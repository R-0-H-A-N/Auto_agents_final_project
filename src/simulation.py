"""Tick-based end-to-end simulation wiring together every module in src/.

Two phases:
    plan_missions  - per bot: LLM dispatch -> route (Dijkstra on fuel) ->
                     Z3 verification of the *real* route (with LLM replan on
                     UNSAT). Then airlock reservations are scheduled for all
                     accepted bots at once.
    execute        - tick loop: bots hold at airlocks until their reservation
                     starts, the adaptive pilot picks thrust on cluttered
                     corridors, the Kalman filter dead-reckons and re-fixes on
                     docking, fuel burns 1 per moving tick.

Run: .venv/bin/python src/simulation.py   (from the project root)
"""

import json
import random
import types
from dataclasses import dataclass, field
from unittest.mock import patch

import networkx as nx
import numpy as np

import dispatcher
from adaptive_pilot import ACTIONS, STATES, QTable, simulate_outcome
from db import (
    get_bot_status,
    get_connection,
    init_db,
    log_event,
    log_task,
    upsert_bot_status,
    upsert_module_inventory,
)
from perception import BotPerception
from routing import edge_key, find_path, path_airlocks, path_fuel_cost, schedule_airlocks
from station import build_default_station
from verifier import verify_task

EPS = 1e-9
TRAIN_EPISODES = 500

# Both bots cross the second airlock (observation_deck-airlock_2_bay) in
# opposite directions at overlapping times, mid-journey. The freeform text
# matches `args` so a live LLM and the offline mock agree on the request.
BOT_IDS = ["bot_1", "bot_2"]
SCENARIOS = {
    "bot_1": {
        "text": "Move 50 kg of cargo from module_b to comms_module",
        "args": {"bot_id": "bot_1", "cargo_mass": 50, "origin": "module_b", "destination": "comms_module"},
        "priority": 5,
        "fuel": 80,
    },
    "bot_2": {
        "text": "Move 30 kg of cargo from module_c to airlock_2_bay",
        "args": {"bot_id": "bot_2", "cargo_mass": 30, "origin": "module_c", "destination": "airlock_2_bay"},
        "priority": 2,
        "fuel": 50,
    },
}

# Starting cargo stock per module (kg); anything unlisted starts empty. Gives
# the Z3 mass constraint something real to check against.
INITIAL_MASSES = {"module_b": 300, "module_c": 400, "comms_module": 850, "airlock_2_bay": 960}


@dataclass
class TickSnapshot:
    tick: int
    bot_positions: dict[str, dict]   # bot_id -> {"edge": (u, v), "true_pos", "believed_pos", "variance", "waiting": bool, "fuel", "state": "moving"|"waiting", "thrust": "normal"|"cautious"|None}
    events: list[dict]               # {"bot": bot_id, "kind": wait|pilot|move|dock|airlock|done, "text": str}
    airlocks: dict[str, str | None]  # "u-v" airlock key -> bot_id using it this tick (or None)


@dataclass
class SimulationTrace:
    station: nx.Graph
    dispatch_log: dict[str, list[dict]]   # bot_id -> attempt_log from dispatch_with_replan
    paths: dict[str, list[str]]           # accepted bots only: bot_id -> route (module names)
    route_info: dict[str, dict]           # bot_id -> {"fuel_cost", "airlocks": ["u-v", ...]}
    schedule: dict                        # routing.schedule_airlocks result (entry_tick, reservations, decisions, planned_finish)
    rejected: dict[str, str]              # bot_id -> final verifier reason (no accepted request)
    ticks: list[TickSnapshot]
    final_status: dict[str, dict]         # bot_id -> final DB bot_status row
    scenarios: dict                       # bot_id -> scenario dict actually run (incl. freeform "text")
    hazards: set                          # hazard modules the verifier saw
    pilot_policy: dict[str, dict]         # "clutter/drift" -> {action: Q-value} learned before the run


# ---------------------------------------------------------------- planning


def _fake_llm_response(args_dict):
    """Same canned-response shape dispatcher.py's own self-test builds."""
    call = types.SimpleNamespace(function=types.SimpleNamespace(arguments=json.dumps(args_dict)))
    message = types.SimpleNamespace(tool_calls=[call])
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


def dispatch_or_mock(text, bot_id, module_names, world_state, correct_args):
    """Try the real LLM call; on failure (no model access), fall back to a
    mocked _call_llm returning the known-correct structured request, so the
    rest of the pipeline still gets real work to do. The warning is
    intentional -- do not suppress it."""
    try:
        return dispatcher.dispatch_with_replan(text, bot_id, module_names, world_state)
    except Exception as e:
        print(
            f"[dispatcher] Live OpenAI call failed ({e}) — falling back to a mocked "
            "LLM parse for this demo run. Set a working OPENAI_API_KEY with model "
            "access to exercise the real path."
        )
        fake = _fake_llm_response(correct_args)
        with patch.object(dispatcher, "_call_llm", return_value=fake):
            return dispatcher.dispatch_with_replan(text, bot_id, module_names, world_state)


def edge_ranges(graph, path):
    """[(u, v, attrs, start_cumulative, end_cumulative), ...] for a path."""
    ranges = []
    cumulative = 0.0
    for u, v in zip(path, path[1:]):
        attrs = graph.get_edge_data(u, v)
        ranges.append((u, v, attrs, cumulative, cumulative + attrs["weight"]))
        cumulative += attrs["weight"]
    return ranges


def drift_bucket(variance):
    if variance < 0.5:
        return "low"
    if variance < 2.0:
        return "medium"
    return "high"


def train_pilot():
    """Light tabular Q-learning pass; the live loop then exploits it."""
    qtable = QTable(STATES, ACTIONS)
    for _ in range(TRAIN_EPISODES):
        state = random.choice(STATES)
        action = qtable.choose_action(state, epsilon=0.2)
        qtable.update(state, action, simulate_outcome(state, action), random.choice(STATES))
    return qtable


def plan_missions(station, conn, scenarios, hazards, cycling):
    """Dispatch + verify (against the real route) every bot, then schedule
    airlocks for the accepted ones. Returns (requests, paths, route_info,
    dispatch_log, rejected, schedule)."""
    module_names = list(station.nodes)
    masses = {m: INITIAL_MASSES.get(m, 0) for m in module_names}
    routes = {}  # (origin, destination) -> path; each route is computed once

    def route_for(origin, destination):
        if (origin, destination) not in routes:
            routes[(origin, destination)] = find_path(station, origin, destination, avoid=hazards)
        return routes[(origin, destination)]

    requests, paths, route_info, dispatch_log, rejected = {}, {}, {}, {}, {}
    for bot_id, sc in scenarios.items():

        def world_state(request, bot_id=bot_id):
            path = route_for(request["origin"], request["destination"])
            return {
                "modules": {m: {"capacity": station.nodes[m]["capacity"], "current_mass": masses[m]} for m in module_names},
                "route": path,
                "route_airlocks": path_airlocks(station, path),
                "cycling_airlocks": cycling,
                "hazard_modules": hazards,
                "fuel_available": sc["fuel"],
                "route_fuel_cost": path_fuel_cost(station, path),
            }

        request, result, attempt_log = dispatch_or_mock(sc["text"], bot_id, module_names, world_state, sc["args"])
        dispatch_log[bot_id] = attempt_log
        if request is None or not result.ok:
            rejected[bot_id] = result.reason
            log_task(conn, bot_id, attempt_log[-1]["request"], outcome="rejected", reason=result.reason)
            continue

        path = route_for(request["origin"], request["destination"])
        requests[bot_id] = request
        paths[bot_id] = path
        route_info[bot_id] = {
            "fuel_cost": path_fuel_cost(station, path),
            "airlocks": ["-".join(e) for e in path_airlocks(station, path)],
        }
        # Commit the cargo so the next bot's mass check sees it.
        masses[request["destination"]] += request["cargo_mass"]
        masses[request["origin"]] = max(0, masses[request["origin"]] - request["cargo_mass"])
        log_task(conn, bot_id, request, outcome="accepted", reason=result.reason)

    plans = [
        {"bot_id": b, "path": paths[b], "priority": scenarios[b]["priority"], "fuel": scenarios[b]["fuel"]}
        for b in paths
    ]
    schedule = schedule_airlocks(station, plans)
    return requests, paths, route_info, dispatch_log, rejected, schedule, masses


# --------------------------------------------------------------- execution


@dataclass
class _Bot:
    bot_id: str
    ranges: list
    entry: dict                       # edge index -> earliest tick to enter (airlocks)
    fuel: float
    perception: BotPerception = field(default_factory=lambda: BotPerception(initial_position=0.0, initial_variance=1.0))
    progress: float = 0.0
    idx: int = 0
    done: bool = False


def execute(station, conn, bots, qtable, max_ticks):
    """Run the tick loop; returns the list of TickSnapshots."""
    ticks = []
    for tick in range(1, max_ticks + 1):
        if all(b.done for b in bots):
            break
        clock = tick - 1  # ticks fully elapsed before this one
        events, positions = [], {}
        airlock_use = {}

        for bot in bots:
            if bot.done:
                continue
            u, v, attrs, start, end = bot.ranges[bot.idx]
            key = edge_key(u, v)
            thrust = None

            at_start = bot.progress <= start + EPS
            if attrs.get("airlock") and at_start and clock < bot.entry[bot.idx]:
                state = "waiting"
                events.append({"bot": bot.bot_id, "kind": "wait",
                               "text": f"[{bot.bot_id}] holding at {u} for airlock {u}-{v} (slot opens at tick {bot.entry[bot.idx]:.0f})"})
            else:
                state = "moving"
                velocity = 1.0
                if attrs.get("cluttered"):
                    bucket = drift_bucket(bot.perception.variance)
                    thrust = qtable.choose_action(("cluttered", bucket), epsilon=0.1)
                    velocity = 0.5 if thrust == "cautious" else 1.0
                    events.append({"bot": bot.bot_id, "kind": "pilot",
                                   "text": f"[{bot.bot_id}] cluttered corridor {u}->{v}, drift risk={bucket} -> pilot chooses '{thrust}' thrust"})
                if attrs.get("airlock"):
                    assert airlock_use.get(key) in (None, bot.bot_id), f"airlock {key} double-booked at tick {tick}"
                    airlock_use[key] = bot.bot_id

                bot.perception.predict(velocity)
                bot.progress += velocity
                bot.fuel = max(0.0, bot.fuel - 1.0)
                events.append({"bot": bot.bot_id, "kind": "move",
                               "text": f"[{bot.bot_id}] believed={bot.perception.position:.3f} variance={bot.perception.variance:.4f} "
                                       f"(progress={bot.progress:.2f}/{bot.ranges[-1][4]:.0f})"})

                if bot.progress >= end - EPS:
                    fix, _ = bot.perception.update(end)
                    events.append({"bot": bot.bot_id, "kind": "dock",
                                   "text": f"[{bot.bot_id}] docked at {v}; fiducial fix -> position={fix:.3f} variance={bot.perception.variance:.4f}"})
                    log_event(conn, "docked", {"bot_id": bot.bot_id, "module": v, "tick": tick})
                    if attrs.get("airlock"):
                        log_event(conn, "airlock_cycle", {"bot_id": bot.bot_id, "edge": [u, v], "tick": tick})
                        events.append({"bot": bot.bot_id, "kind": "airlock", "text": f"[{bot.bot_id}] crossed the airlock {u}-{v}"})
                    bot.idx += 1
                    if bot.idx >= len(bot.ranges):
                        bot.done = True
                        events.append({"bot": bot.bot_id, "kind": "done", "text": f"[{bot.bot_id}] reached final destination {v}, task complete"})

            positions[bot.bot_id] = {
                "edge": (u, v),
                "true_pos": bot.progress,
                "believed_pos": bot.perception.position,
                "variance": bot.perception.variance,
                "waiting": state == "waiting",
                "fuel": bot.fuel,
                "state": state,
                "thrust": thrust,
            }

        airlocks = {
            "-".join(edge_key(a, b)): airlock_use.get(edge_key(a, b))
            for a, b, d in station.edges(data=True) if d.get("airlock")
        }
        ticks.append(TickSnapshot(tick=tick, bot_positions=positions, events=events, airlocks=airlocks))
    return ticks


def run_simulation(scenarios: dict | None = None, hazards: set | None = None, cycling_airlocks: set | None = None) -> SimulationTrace:
    random.seed(0)
    np.random.seed(0)  # Kalman drift noise: reproducible runs

    station = build_default_station()
    conn = get_connection()
    init_db(conn)

    scenarios = scenarios or SCENARIOS
    hazards = set(hazards or ())
    cycling = {edge_key(*e) for e in (cycling_airlocks or ())}

    requests, paths, route_info, dispatch_log, rejected, schedule, masses = plan_missions(
        station, conn, scenarios, hazards, cycling
    )

    qtable = train_pilot()
    bots = [
        _Bot(b, edge_ranges(station, paths[b]), schedule["entry_tick"][b], float(scenarios[b]["fuel"]))
        for b in paths
    ]
    # Worst-case plan length plus slack: the loop normally exits when all bots finish.
    max_ticks = int(max(schedule["planned_finish"].values(), default=0)) + 5
    ticks = execute(station, conn, bots, qtable, max_ticks)
    assert all(b.done for b in bots), "simulation ended with unfinished bots -- worst-case timing was violated"

    # Persist outcomes: real depleted fuel, final module masses.
    for bot in bots:
        upsert_bot_status(conn, bot.bot_id, fuel=bot.fuel, believed_position=requests[bot.bot_id]["destination"], current_task="idle")
        log_task(conn, bot.bot_id, requests[bot.bot_id], outcome="completed")
    for module, mass in masses.items():
        upsert_module_inventory(conn, module, mass=mass)

    final_status = {b.bot_id: get_bot_status(conn, b.bot_id) for b in bots}
    conn.close()

    return SimulationTrace(
        station=station,
        dispatch_log=dispatch_log,
        paths=paths,
        route_info=route_info,
        schedule=schedule,
        rejected=rejected,
        ticks=ticks,
        final_status=final_status,
        scenarios=scenarios,
        hazards=hazards,
        pilot_policy={f"{c}/{d}": dict(qtable.q[(c, d)]) for c, d in STATES},
    )


def print_trace(trace: SimulationTrace):
    print("=" * 70)
    print("PLAN: dispatch -> route -> verify (real route)")
    print("=" * 70)
    for bot_id, attempt_log in trace.dispatch_log.items():
        print(f"\n[{bot_id}] command: {trace.scenarios[bot_id]['text']!r}")
        for entry in attempt_log:
            print(f"[{bot_id}] attempt {entry['attempt']}: {entry['request']} -> {entry['verification']}")
        if bot_id in trace.rejected:
            print(f"[{bot_id}] REJECTED: {trace.rejected[bot_id]}")
        else:
            info = trace.route_info[bot_id]
            print(f"[{bot_id}] route: {trace.paths[bot_id]} (fuel cost {info['fuel_cost']}, airlocks {info['airlocks'] or 'none'})")

    print("\nairlock schedule:")
    if not trace.schedule["decisions"]:
        print("  no overlapping airlock use - nobody waits")
    for d in trace.schedule["decisions"]:
        print(f"  {d['airlock']}: {d['winner']} first ({d['reason']}); {d['loser']} waits {d['wait_ticks']:.0f} tick(s)")

    print("\n" + "=" * 70)
    print("TICK LOOP")
    print("=" * 70)
    for tick in trace.ticks:
        print(f"\n-- tick {tick.tick} --")
        for e in tick.events:
            print(e["text"])

    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    for bot_id, status in trace.final_status.items():
        print(f"[{bot_id}] DB bot_status readback: {status}")
    print("\nsimulation complete.")


def main():
    print_trace(run_simulation())


if __name__ == "__main__":
    main()

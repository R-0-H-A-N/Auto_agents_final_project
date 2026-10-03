"""Pathfinding and multi-airlock scheduling over the station graph.

Two jobs:
    1. find_path        - fuel-cheapest route (cluttered corridors cost 2x).
    2. schedule_airlocks - reservation table per airlock; a priority rule
       decides who goes first, and everyone else is delayed *at the node
       before the airlock* until the lane is free.

Time model: edge durations in the schedule are worst case (cluttered edges
counted at cautious speed, i.e. the fuel weight), so a bot can only ever
arrive at an airlock early, never late. At runtime a bot that arrives early
holds until its reservation starts, which keeps reservations conflict-free
no matter what the adaptive pilot chooses.
"""

import networkx as nx

from station import build_default_station


def _fuel_weight(u, v, attrs: dict) -> float:
    """Fuel cost of an edge: cluttered corridors cost 2x because the adaptive
    pilot takes cautious (half-speed) thrust there, doubling ticks spent and
    therefore fuel burned (see simulation.py's per-tick fuel deduction). This
    doubles as the *worst-case* traversal time in ticks."""
    return attrs["weight"] * (2 if attrs.get("cluttered") else 1)


def edge_key(u, v) -> tuple:
    """Direction-agnostic key: a single-lane airlock blocks both directions."""
    return tuple(sorted((u, v)))


def find_path(graph: nx.Graph, origin: str, destination: str, avoid=frozenset()) -> list:
    """Fuel-cheapest path. With no coordinates there is no useful A* heuristic,
    so this is Dijkstra on the fuel weight (not raw distance): bots minimise
    fuel burn, which only differs from distance when clutter is involved.

    `avoid` (hazard modules) is detoured around when a detour exists; if none
    does, the unrestricted path is returned and the Z3 verifier rejects it."""
    allowed = graph.subgraph(n for n in graph if n not in avoid or n in (origin, destination))
    try:
        return nx.dijkstra_path(allowed, origin, destination, weight=_fuel_weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return nx.dijkstra_path(graph, origin, destination, weight=_fuel_weight)


def path_fuel_cost(graph: nx.Graph, path: list) -> float:
    """Total fuel cost of an already-computed path."""
    return sum(_fuel_weight(u, v, graph.get_edge_data(u, v)) for u, v in zip(path, path[1:]))


def path_airlocks(graph: nx.Graph, path: list) -> list:
    """Airlock edges (as (u, v) in travel order) that a path crosses."""
    return [(u, v) for u, v in zip(path, path[1:]) if graph.get_edge_data(u, v).get("airlock")]


def _priority_order(plans: list) -> list:
    """Higher priority first; tie broken by lower fuel; then bot_id for stability."""
    return sorted(plans, key=lambda p: (-p["priority"], p["fuel"], p["bot_id"]))


def _why_first(first: dict, second: dict) -> str:
    if first["priority"] != second["priority"]:
        return f"higher priority ({first['priority']} > {second['priority']})"
    if first["fuel"] != second["fuel"]:
        return f"equal priority, lower fuel ({first['fuel']} < {second['fuel']})"
    return "equal priority and fuel, stable bot_id order"


def schedule_airlocks(graph: nx.Graph, plans: list) -> dict:
    """Greedy reservation scheduling, highest-priority bot first.

    plans: [{bot_id, path, priority, fuel}, ...]
    Returns {
        "entry_tick": {bot_id: {edge_index_in_path: earliest_tick_to_enter}},
        "reservations": {airlock_key: [(bot_id, start, end), ...]},
        "decisions": [{airlock, winner, loser, wait_ticks, reason}, ...],
        "planned_finish": {bot_id: tick},       # worst-case completion tick
    }
    Every airlock crossing gets an entry tick (not only delayed ones), so a bot
    that arrives early still waits for its slot.
    """
    entry_tick = {p["bot_id"]: {} for p in plans}
    reservations = {}
    decisions = []
    planned_finish = {}
    by_id = {p["bot_id"]: p for p in plans}

    for plan in _priority_order(plans):
        clock = 0.0
        for idx, (u, v) in enumerate(zip(plan["path"], plan["path"][1:])):
            attrs = graph.get_edge_data(u, v)
            duration = _fuel_weight(u, v, attrs)
            if attrs.get("airlock"):
                key = edge_key(u, v)
                start = clock
                blockers = set()
                moved = True
                while moved:  # push start past every overlapping reservation
                    moved = False
                    for other, r_start, r_end in reservations.get(key, []):
                        if start < r_end and r_start < start + duration:
                            start = r_end
                            blockers.add(other)
                            moved = True
                if start > clock:
                    winner = _priority_order([by_id[b] for b in blockers])[0]
                    decisions.append({
                        "airlock": f"{key[0]}-{key[1]}",
                        "winner": winner["bot_id"],
                        "loser": plan["bot_id"],
                        "wait_ticks": start - clock,
                        "reason": _why_first(winner, plan),
                    })
                reservations.setdefault(key, []).append((plan["bot_id"], start, start + duration))
                entry_tick[plan["bot_id"]][idx] = start
                clock = start
            clock += duration
        planned_finish[plan["bot_id"]] = clock

    return {
        "entry_tick": entry_tick,
        "reservations": reservations,
        "decisions": decisions,
        "planned_finish": planned_finish,
    }


if __name__ == "__main__":
    station = build_default_station()

    # Fuel-aware routing: module_b-module_c is cluttered (weight 3 -> 6 fuel),
    # module_c-module_e is not (weight 2).
    assert path_fuel_cost(station, ["module_b", "module_c", "module_e"]) == 3 * 2 + 2

    # Case 1: opposite directions through the same airlock, decided by priority.
    p1 = find_path(station, "docking_bay", "module_b")
    p2 = find_path(station, "module_a", "docking_bay")
    bot_1 = {"bot_id": "bot_1", "path": p1, "priority": 5, "fuel": 80}
    bot_2 = {"bot_id": "bot_2", "path": p2, "priority": 2, "fuel": 20}
    sched = schedule_airlocks(station, [bot_1, bot_2])
    print("priority case:", sched["decisions"])
    assert len(sched["decisions"]) == 1
    assert sched["decisions"][0]["winner"] == "bot_1" and sched["decisions"][0]["loser"] == "bot_2"
    assert sched["decisions"][0]["wait_ticks"] == 1

    # Case 2: equal priority -> lower fuel wins.
    bot_3 = {"bot_id": "bot_3", "path": p1, "priority": 3, "fuel": 15}
    bot_4 = {"bot_id": "bot_4", "path": p2, "priority": 3, "fuel": 90}
    sched = schedule_airlocks(station, [bot_4, bot_3])
    assert sched["decisions"][0]["winner"] == "bot_3"

    # Case 3: reservations never overlap on a shared airlock.
    for key, slots in sched["reservations"].items():
        slots = sorted(slots, key=lambda s: s[1])
        assert all(a[2] <= b[1] for a, b in zip(slots, slots[1:])), key

    # Case 4: the SECOND airlock is arbitrated too (old code only saw the first).
    q1 = ["airlock_2_bay", "observation_deck"]
    q2 = ["observation_deck", "airlock_2_bay"]
    sched = schedule_airlocks(
        station,
        [{"bot_id": "a", "path": q1, "priority": 1, "fuel": 10},
         {"bot_id": "b", "path": q2, "priority": 9, "fuel": 10}],
    )
    assert sched["decisions"][0]["airlock"] == "airlock_2_bay-observation_deck"
    assert sched["decisions"][0]["winner"] == "b"

    # Case 5: no shared airlock -> nobody waits.
    sched = schedule_airlocks(
        station,
        [{"bot_id": "a", "path": ["docking_bay", "module_a"], "priority": 1, "fuel": 1},
         {"bot_id": "b", "path": ["observation_deck", "airlock_2_bay"], "priority": 1, "fuel": 1}],
    )
    assert sched["decisions"] == []

    # Hazard avoidance: module_b is skipped via module_d -> cargo_bay -> module_c.
    assert "module_b" not in find_path(station, "module_a", "module_c", avoid={"module_b"})
    assert "module_b" in find_path(station, "module_a", "module_c")

    # Case 6: waiting happens at the airlock, AFTER a cluttered edge slowed the
    # bot -- the planned entry tick uses worst-case (cautious) timing.
    path = ["solar_array", "comms_module", "observation_deck", "airlock_2_bay"]
    sched = schedule_airlocks(station, [{"bot_id": "a", "path": path, "priority": 1, "fuel": 1}])
    assert sched["entry_tick"]["a"][2] == 2 * 2 + 2  # cluttered edge (w=2) counted at 2x, then plain edge w=2

    print("all assertions passed")

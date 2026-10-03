"""SMT-based safety verifier for cargo-bot task requests.

Standalone and decoupled: takes plain dicts for request/world_state, does not
import station.py or db.py. Uses Z3 to check safety constraints against the
*actual planned route* (the caller routes first, then verifies):

    mass_limit            destination has room for the cargo
    airlock_state         no airlock on the route is mid-cycle
    hazard_origin/dest    origin / destination are not hazard zones
    hazard_route          no module the route passes through is a hazard zone
    fuel_budget           the bot has enough fuel for the route's fuel cost

On UNSAT every violated constraint is reported (not just the first), via
repeated unsat-core extraction, so the LLM replan loop sees the full picture.
"""

from dataclasses import dataclass

import z3


@dataclass
class VerificationResult:
    ok: bool
    reason: str = "all constraints satisfied"


def _edge_key(u, v):
    return tuple(sorted((u, v)))


def verify_task(request, world_state):
    """request: {bot_id, cargo_mass, origin, destination}
    world_state: {
        modules: {name: {"capacity": float, "current_mass": float}},
        route: [module, ...],                  # planned path (default: origin, destination)
        route_airlocks: [(u, v), ...],         # airlock edges the route crosses
        cycling_airlocks: {(u, v), ...},       # airlocks currently mid-cycle
        hazard_modules: set(str),
        fuel_available: float | None,          # optional; enables fuel_budget
        route_fuel_cost: float | None,
    }
    """
    origin = request["origin"]
    destination = request["destination"]
    cargo_mass = request["cargo_mass"]

    dest_info = world_state["modules"][destination]
    route = world_state.get("route") or [origin, destination]
    route_airlocks = [_edge_key(*e) for e in world_state.get("route_airlocks", [])]
    cycling = {_edge_key(*e) for e in world_state.get("cycling_airlocks", set())}
    hazards = world_state.get("hazard_modules", set())
    fuel_available = world_state.get("fuel_available")
    route_fuel_cost = world_state.get("route_fuel_cost")

    # name -> (z3 constraint, human-readable reason when violated)
    constraints = {
        "mass_limit": (
            z3.RealVal(dest_info["current_mass"]) + z3.RealVal(cargo_mass) <= z3.RealVal(dest_info["capacity"]),
            f"mass limit exceeded at {destination}: {dest_info['current_mass']}kg + {cargo_mass}kg = "
            f"{dest_info['current_mass'] + cargo_mass}kg > {dest_info['capacity']}kg capacity",
        ),
        "airlock_state": (
            z3.Not(z3.BoolVal(any(e in cycling for e in route_airlocks))),
            "airlock is mid-cycle, cannot route through "
            + ", ".join("-".join(e) for e in route_airlocks if e in cycling),
        ),
        "hazard_origin": (
            z3.BoolVal(origin not in hazards),
            f"hazard zone active at {origin}, origin is off-limits",
        ),
        "hazard_destination": (
            z3.BoolVal(destination not in hazards),
            f"hazard zone active at {destination}, destination is off-limits",
        ),
        "hazard_route": (
            z3.BoolVal(not any(m in hazards for m in route[1:-1])),
            "route passes through hazard zone(s): " + ", ".join(m for m in route[1:-1] if m in hazards),
        ),
    }
    if fuel_available is not None and route_fuel_cost is not None:
        constraints["fuel_budget"] = (
            z3.RealVal(route_fuel_cost) <= z3.RealVal(fuel_available),
            f"insufficient fuel: route needs {route_fuel_cost} but bot has {fuel_available}",
        )

    # Extract unsat cores repeatedly, dropping each found core, to list every
    # violated constraint rather than only the first one Z3 happens to report.
    active = dict(constraints)
    violated = []
    while True:
        solver = z3.Solver()
        for name, (expr, _) in active.items():
            solver.assert_and_track(expr, name)
        if solver.check() == z3.sat:
            break
        core = {str(c) for c in solver.unsat_core()}
        violated += [n for n in active if n in core]
        active = {n: c for n, c in active.items() if n not in core}

    if not violated:
        return VerificationResult(True)
    return VerificationResult(False, "; ".join(constraints[n][1] for n in violated))


def _demo():
    base_modules = {
        "docking_bay": {"capacity": 5000, "current_mass": 100},
        "module_a": {"capacity": 2000, "current_mass": 500},
        "module_b": {"capacity": 2000, "current_mass": 0},
        "cargo_bay": {"capacity": 8000, "current_mass": 7950},
        "module_c": {"capacity": 1500, "current_mass": 200},
    }

    def make_world(**overrides):
        world = {
            "modules": base_modules,
            "route": ["docking_bay", "module_a"],
            "route_airlocks": [],
            "cycling_airlocks": set(),
            "hazard_modules": set(),
        }
        world.update(overrides)
        return world

    request = {"bot_id": "bot_1", "cargo_mass": 50, "origin": "docking_bay", "destination": "module_a"}

    result = verify_task(request, make_world())
    print("SAT case:", result)
    assert result.ok

    big = {"bot_id": "bot_1", "cargo_mass": 500, "origin": "module_c", "destination": "cargo_bay"}
    result = verify_task(big, make_world())
    print("UNSAT mass case:", result)
    assert not result.ok and "mass limit exceeded" in result.reason

    # Airlock only matters if the route crosses it AND it is cycling.
    airlock = ("docking_bay", "module_a")
    assert verify_task(request, make_world(cycling_airlocks={airlock})).ok
    result = verify_task(request, make_world(route_airlocks=[airlock], cycling_airlocks={("module_a", "docking_bay")}))
    print("UNSAT airlock case:", result)
    assert not result.ok and "mid-cycle" in result.reason

    result = verify_task(request, make_world(hazard_modules={"module_a"}))
    print("UNSAT hazard case:", result)
    assert not result.ok and "hazard zone active" in result.reason

    # Hazard on an intermediate module is caught only because we verify the real route.
    through = make_world(route=["docking_bay", "module_b", "module_a"], hazard_modules={"module_b"})
    result = verify_task(request, through)
    print("UNSAT hazard-on-route case:", result)
    assert not result.ok and "passes through" in result.reason

    result = verify_task(request, make_world(fuel_available=5, route_fuel_cost=9))
    print("UNSAT fuel case:", result)
    assert not result.ok and "insufficient fuel" in result.reason

    # Multiple violations are all reported.
    result = verify_task(big, make_world(hazard_modules={"module_c"}))
    print("multi-violation case:", result)
    assert "mass limit" in result.reason and "hazard zone" in result.reason

    print("self-test passed")


if __name__ == "__main__":
    _demo()

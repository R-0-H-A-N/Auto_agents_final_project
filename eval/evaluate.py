"""Evaluation (offline-safe: no OpenAI key needed).

1. Verifier accuracy on a labelled set of safe / unsafe requests.
2. Metrics from the default simulation.

Run from the project root:
    .venv/bin/python eval/evaluate.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from unittest.mock import patch

import dispatcher
import simulation
from test_core import _mock_llm
from verifier import verify_task

MODULES = {
    "module_a": {"capacity": 2000, "current_mass": 500},
    "module_b": {"capacity": 2000, "current_mass": 0},
    "cargo_bay": {"capacity": 8000, "current_mass": 7950},
}


def _world(**kw):
    w = {"modules": MODULES, "route": ["module_a", "module_b"], "route_airlocks": [],
         "cycling_airlocks": set(), "hazard_modules": set()}
    return {**w, **kw}


def _req(mass=50, dest="module_b"):
    return {"bot_id": "bot_1", "cargo_mass": mass, "origin": "module_a", "destination": dest}


# (request, world_state, expected_ok)
CASES = [
    (_req(), _world(), True),
    (_req(mass=1500), _world(), True),
    (_req(), _world(route_fuel_cost=5, fuel_available=10), True),
    (_req(mass=5000), _world(), False),                                       # mass
    (_req(mass=100, dest="cargo_bay"), _world(route=["module_a", "cargo_bay"]), False),  # mass at cargo_bay
    (_req(), _world(hazard_modules={"module_b"}), False),                      # hazard at destination
    (_req(), _world(hazard_modules={"module_a"}), False),                      # hazard at origin
    (_req(), _world(route_airlocks=[("module_a", "module_b")],
                    cycling_airlocks={("module_a", "module_b")}), False),      # cycling airlock
    (_req(), _world(route_fuel_cost=9, fuel_available=5), False),              # fuel
]


def main():
    correct = sum(verify_task(r, w).ok == exp for r, w, exp in CASES)
    metrics = {"verifier_accuracy": correct / len(CASES), "verifier_cases": len(CASES)}
    print(f"verifier: {correct}/{len(CASES)} correct")

    with patch.object(dispatcher, "_call_llm", _mock_llm):
        trace = simulation.run_simulation()
    metrics["sim_finish_tick"] = trace.ticks[-1].tick
    metrics["sim_rejected"] = len(trace.rejected)
    print(metrics)

    assert correct == len(CASES), "verifier disagreed with a labelled case"


if __name__ == "__main__":
    main()

"""Offline checks: no API key or network needed (the LLM call is patched).

Run from the project root:  .venv/bin/python tests/test_core.py
"""

import json
import os
import sys
import types
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dispatcher
import simulation
import verifier
from routing import find_path, path_airlocks
from station import build_default_station


def _mock_llm(messages, tools):
    """Answer each command with that scenario's known-correct request."""
    text = messages[1]["content"]  # later messages are replan feedback
    args = next(s["args"] for s in simulation.SCENARIOS.values() if s["text"] == text)
    call = types.SimpleNamespace(function=types.SimpleNamespace(arguments=json.dumps(args)))
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(tool_calls=[call]))])


def test_verifier_branches():
    verifier._demo()  # asserts SAT, mass, airlock, hazard and fuel cases


def test_station_shape():
    g = build_default_station()
    assert g.number_of_nodes() == 17 and g.number_of_edges() == 20
    assert len(path_airlocks(g, find_path(g, "module_b", "comms_module"))) >= 1


def test_hazard_detour():
    g = build_default_station()
    direct = find_path(g, "module_b", "comms_module")
    detour = find_path(g, "module_b", "comms_module", avoid=frozenset({direct[1]}))
    assert direct[1] not in detour


def test_default_simulation():
    with patch.object(dispatcher, "_call_llm", _mock_llm):
        trace = simulation.run_simulation()
    assert not trace.rejected
    assert {b["fuel"] for b in trace.final_status.values()} == {64.0, 35.0}
    for tick in trace.ticks:  # one bot per airlock per tick is enforced by execute(); re-check
        assert all(v is None or isinstance(v, str) for v in tick.airlocks.values())
    assert trace.ticks[-1].tick == 16


def test_cycling_airlock_rejected():
    with patch.object(dispatcher, "_call_llm", _mock_llm):
        trace = simulation.run_simulation(cycling_airlocks={("observation_deck", "airlock_2_bay")})
    assert "bot_2" in trace.rejected or "bot_1" in trace.rejected


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)

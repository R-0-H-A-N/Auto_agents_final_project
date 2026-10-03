"""Freeform-text -> structured cargo task via OpenAI function-calling, plus a
reflect-and-replan loop against verifier.verify_task.

The real OpenAI call is isolated in `_call_llm` so it can be monkeypatched in
tests without hitting the network.
"""

import json
import os
from dataclasses import asdict

from dotenv import load_dotenv
from openai import OpenAI

from verifier import verify_task

load_dotenv()

MODEL = "gpt-5-mini"
FUNCTION_NAME = "dispatch_cargo_task"


def _tool_schema(bot_id, module_names):
    """bot_id's enum is pinned to a single value so the LLM cannot assign the
    task to a different bot than the one this command is being dispatched
    for (there is no other signal telling it which bot is asking)."""
    return [
        {
            "type": "function",
            "function": {
                "name": FUNCTION_NAME,
                "description": "Dispatch a cargo-moving task to a bot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bot_id": {"type": "string", "enum": [bot_id]},
                        "cargo_mass": {"type": "number"},
                        "origin": {"type": "string", "enum": module_names},
                        "destination": {"type": "string", "enum": module_names},
                    },
                    "required": ["bot_id", "cargo_mass", "origin", "destination"],
                },
            },
        }
    ]


def _call_llm(messages, tools):
    """Thin wrapper around the OpenAI chat completions call. Isolated so
    tests can monkeypatch this one function instead of hitting the network."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": FUNCTION_NAME}},
    )


def _extract_request(response):
    call = response.choices[0].message.tool_calls[0]
    return json.loads(call.function.arguments)


def parse_command(text: str, bot_id: str, module_names: list) -> dict:
    """Turn a freeform text command into a structured request dict matching
    verifier.py's expected shape: {bot_id, cargo_mass, origin, destination}."""
    tools = _tool_schema(bot_id, module_names)
    messages = [
        {
            "role": "system",
            "content": (
                f"You translate freeform cargo-move commands into a structured "
                f"call to {FUNCTION_NAME}. This command is being dispatched for "
                f"bot {bot_id}; always use that as bot_id."
            ),
        },
        {"role": "user", "content": text},
    ]
    response = _call_llm(messages, tools)
    return _extract_request(response)


def dispatch_with_replan(text, bot_id, module_names, world_state, max_attempts=3):
    """Parse `text` into a request, verify it, and if UNSAT, feed the
    rejection reason back to the LLM and ask for a revised proposal, up to
    `max_attempts` times.

    `world_state` is either a dict or a callable `request -> dict`. The
    callable form lets the caller route each proposal first, so the verifier
    sees the real route (airlocks crossed, fuel cost) instead of guesses.

    Returns (final_request_or_None, verification_result, attempt_log).
    """
    tools = _tool_schema(bot_id, module_names)
    messages = [
        {
            "role": "system",
            "content": (
                f"You translate freeform cargo-move commands into a structured "
                f"call to {FUNCTION_NAME}. This command is being dispatched for "
                f"bot {bot_id}; always use that as bot_id. If a previous "
                "proposal was rejected for safety reasons, propose a safe "
                "alternative (different destination, reduced cargo_mass, etc.) "
                "or state there is none."
            ),
        },
        {"role": "user", "content": text},
    ]

    attempt_log = []
    last_result = None

    for attempt in range(1, max_attempts + 1):
        response = _call_llm(messages, tools)
        request = _extract_request(response)
        result = verify_task(request, world_state(request) if callable(world_state) else world_state)

        attempt_log.append(
            {"attempt": attempt, "request": request, "verification": asdict(result)}
        )

        if result.ok:
            return request, result, attempt_log

        last_result = result
        messages.append(
            {
                "role": "assistant",
                "content": f"Proposed: {json.dumps(request)}",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Your previous plan was rejected: {result.reason}. Propose "
                    "an alternative structured request - e.g. a different "
                    "destination, a reduced cargo_mass, or note if there's no "
                    "safe alternative."
                ),
            }
        )

    return None, last_result, attempt_log


if __name__ == "__main__":
    import sys
    import types
    from unittest.mock import patch

    _this_module = sys.modules[__name__]

    def fake_response(args_dict):
        call = types.SimpleNamespace(
            function=types.SimpleNamespace(arguments=json.dumps(args_dict))
        )
        message = types.SimpleNamespace(tool_calls=[call])
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])

    world_state = {
        "modules": {
            "docking_bay": {"capacity": 5000, "current_mass": 100},
            "module_a": {"capacity": 2000, "current_mass": 500},
            "cargo_bay": {"capacity": 8000, "current_mass": 7950},
        },
        "hazard_modules": set(),
    }
    bot_id = "bot_1"
    module_names = ["docking_bay", "module_a", "cargo_bay"]

    # First call proposes an overloaded destination (UNSAT), second call
    # replans to a destination with room (SAT).
    responses = [
        fake_response(
            {"bot_id": "bot_1", "cargo_mass": 500, "origin": "docking_bay", "destination": "cargo_bay"}
        ),
        fake_response(
            {"bot_id": "bot_1", "cargo_mass": 500, "origin": "docking_bay", "destination": "module_a"}
        ),
    ]

    with patch.object(_this_module, "_call_llm", side_effect=responses):
        # parse_command exercised standalone
        req = parse_command("Move O2 canister from docking bay to module A", bot_id, module_names)
        assert req["destination"] == "cargo_bay"

    with patch.object(_this_module, "_call_llm", side_effect=list(responses)):
        request, result, log = dispatch_with_replan(
            "Move O2 canister from docking bay to cargo bay", bot_id, module_names, world_state
        )
        print("final request:", request)
        print("final result:", result)
        print("attempt log:", log)
        assert result.ok
        assert request["destination"] == "module_a"
        assert len(log) == 2
        assert not log[0]["verification"]["ok"]
        assert log[1]["verification"]["ok"]

    print("self-test passed")

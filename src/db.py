"""Persistence layer for the orbital cargo-bot simulation.

Plain sqlite3, stdlib only. No ORM, no migrations framework.

Tables: module_inventory (module mass), bot_status (fuel, believed position,
current task), task_history (requests with outcome accepted/rejected/completed
and reason), events (docking and airlock-cycle log). simulation.py writes
masses and fuel at the end of a run and tasks/events as they happen.
"""

import json
import os
import sqlite3
import time

# Anchored to this file, not the cwd, so the DB opens from anywhere.
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbital.db")


def get_connection(db_path=DEFAULT_DB_PATH):
    """Open SQLite with dict-like rows. Pass \":memory:\" for a throwaway DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    """Create all four tables if missing (idempotent)."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS module_inventory (
            module_id TEXT PRIMARY KEY,
            mass REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bot_status (
            bot_id TEXT PRIMARY KEY,
            fuel REAL NOT NULL,
            believed_position TEXT,
            current_task TEXT
        );

        CREATE TABLE IF NOT EXISTS task_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id TEXT NOT NULL,
            request TEXT NOT NULL,
            timestamp REAL NOT NULL,
            outcome TEXT NOT NULL,
            reason TEXT
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            details TEXT
        );
        """
    )
    conn.commit()


def upsert_module_inventory(conn, module_id, mass):
    """Insert or update a module's cargo mass."""
    conn.execute(
        """
        INSERT INTO module_inventory (module_id, mass) VALUES (?, ?)
        ON CONFLICT(module_id) DO UPDATE SET mass = excluded.mass
        """,
        (module_id, mass),
    )
    conn.commit()


def upsert_bot_status(conn, bot_id, fuel, believed_position, current_task):
    """Insert or update a bot's fuel, believed position and current task."""
    conn.execute(
        """
        INSERT INTO bot_status (bot_id, fuel, believed_position, current_task)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(bot_id) DO UPDATE SET
            fuel = excluded.fuel,
            believed_position = excluded.believed_position,
            current_task = excluded.current_task
        """,
        (bot_id, fuel, believed_position, current_task),
    )
    conn.commit()


def log_task(conn, bot_id, request, outcome, reason=None):
    """`request` may be a dict (will be JSON-encoded) or a string."""
    if not isinstance(request, str):
        request = json.dumps(request)
    conn.execute(
        "INSERT INTO task_history (bot_id, request, timestamp, outcome, reason) "
        "VALUES (?, ?, ?, ?, ?)",
        (bot_id, request, time.time(), outcome, reason),
    )
    conn.commit()


def log_event(conn, event_type, details=None):
    """`details` may be a dict (will be JSON-encoded) or a string."""
    if details is not None and not isinstance(details, str):
        details = json.dumps(details)
    conn.execute(
        "INSERT INTO events (timestamp, event_type, details) VALUES (?, ?, ?)",
        (time.time(), event_type, details),
    )
    conn.commit()


def get_bot_status(conn, bot_id):
    """Bot row as a dict, or None."""
    row = conn.execute(
        "SELECT * FROM bot_status WHERE bot_id = ?", (bot_id,)
    ).fetchone()
    return dict(row) if row else None


def get_module_inventory(conn, module_id):
    """Module row as a dict, or None."""
    row = conn.execute(
        "SELECT * FROM module_inventory WHERE module_id = ?", (module_id,)
    ).fetchone()
    return dict(row) if row else None


if __name__ == "__main__":
    conn = get_connection(":memory:")
    init_db(conn)

    upsert_module_inventory(conn, "module_a", 120.5)
    upsert_bot_status(conn, "bot_1", fuel=88.0, believed_position="module_a", current_task="idle")
    log_task(
        conn,
        "bot_1",
        {"origin": "module_a", "destination": "module_b", "cargo_mass": 30},
        outcome="accepted",
    )
    log_event(conn, "airlock_cycle", {"bot_id": "bot_1", "direction": "out"})

    module = get_module_inventory(conn, "module_a")
    assert module["mass"] == 120.5
    print("module_inventory:", module)

    bot = get_bot_status(conn, "bot_1")
    assert bot["fuel"] == 88.0
    assert bot["believed_position"] == "module_a"
    print("bot_status:", bot)

    task_row = conn.execute("SELECT * FROM task_history").fetchone()
    assert task_row["bot_id"] == "bot_1"
    assert task_row["outcome"] == "accepted"
    print("task_history:", dict(task_row))

    event_row = conn.execute("SELECT * FROM events").fetchone()
    assert event_row["event_type"] == "airlock_cycle"
    print("events:", dict(event_row))

    print("self-test passed")

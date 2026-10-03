"""Streamlit dashboard for the orbital cargo bots simulation.

Runs the simulation once (cached in st.session_state), then scrubs/plays
through the precomputed SimulationTrace: station map with an uncertainty
heatmap, per-bot Gantt timeline with airlock reservations, fuel/variance
charts, status cards, the current tick's algorithm events, and reasoning-log
tabs (dispatch + Z3, airlock schedule, adaptive pilot, event history).

Run: .venv/bin/streamlit run src/dashboard.py   (from the project root)
"""

import time

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from simulation import SCENARIOS, run_simulation

st.set_page_config(page_title="Orbital Cargo Bots", layout="wide", page_icon="🛰️")

BG = "#0F1116"
BOT_COLORS = {}
_PALETTE = ["#2E86AB", "#E67E22", "#27AE60", "#8E44AD", "#C0392B"]

EDGE_STYLE = {
    "airlock": dict(color="#8FA0FF", linestyle="--", linewidth=2.5, label="Airlock (single-lane chokepoint)"),
    "cluttered": dict(color="#C0392B", linestyle="-", linewidth=2.5, label="Cluttered corridor (adaptive thrust)"),
    "plain": dict(color="#B0B7C3", linestyle="-", linewidth=1.5, label="Open corridor"),
}
OUTLINE = [pe.withStroke(linewidth=2.5, foreground=BG)]

# Timeline block colours by (state, thrust).
STATE_COLORS = {
    "normal": "#27AE60",
    "cautious": "#E6B422",
    "waiting": "#C0392B",
}
EVENT_ICONS = {"wait": "⏳", "pilot": "⚠️", "move": "•", "dock": "⚓", "airlock": "🚦", "done": "🏁"}


def bot_color(bot_id):
    """Stable colour per bot id."""
    if bot_id not in BOT_COLORS:
        BOT_COLORS[bot_id] = _PALETTE[len(BOT_COLORS) % len(_PALETTE)]
    return BOT_COLORS[bot_id]


def node_layout(graph):
    """Spring layout, seeded for a fixed, deterministic result every run."""
    return nx.spring_layout(graph, seed=42, weight="weight")


def edge_start_cumulative(graph, path, u):
    """Cumulative path distance at the start of the edge beginning at u."""
    cumulative = 0.0
    for a, b in zip(path, path[1:]):
        if a == u:
            return cumulative
        cumulative += graph.get_edge_data(a, b)["weight"]
    return cumulative


def lerp(p0, p1, frac):
    """Point a fraction `frac` of the way from p0 to p1."""
    return (p0[0] + (p1[0] - p0[0]) * frac, p0[1] + (p1[1] - p0[1]) * frac)


def edge_kind(attrs):
    """'airlock' | 'cluttered' | 'plain' for an edge attribute dict."""
    if attrs.get("airlock"):
        return "airlock"
    if attrs.get("cluttered"):
        return "cluttered"
    return "plain"


def bot_xy(trace, bot_id, pos, coords):
    """(true_xy, believed_xy) of a bot on the map, interpolated along its edge."""
    u, v = pos["edge"]
    weight = trace.station.get_edge_data(u, v)["weight"]
    start = edge_start_cumulative(trace.station, trace.paths[bot_id], u)

    def at(value):
        frac = (value - start) / weight if weight else 0.0
        return lerp(coords[u], coords[v], max(0.0, min(1.0, frac)))

    return at(pos["true_pos"]), at(pos["believed_pos"])


def uncertainty_field(points, bounds, n=100):
    """Gaussian blob per (x, y, variance) summed on an n x n grid.

    Returns the grid; sigma grows with sqrt(variance) so a drifting bot
    glows wider, a freshly docked one collapses to a tight dot.
    """
    x0, x1, y0, y1 = bounds
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    field = np.zeros((n, n))
    for x, y, var in points:
        sigma = 0.05 + 0.08 * np.sqrt(var)
        field += np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sigma**2))
    return field


def draw_map(trace, snapshot, coords):
    """Station map: heatmap, edges, nodes, hazards, planned paths, bots."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    xs = [x for x, _ in coords.values()]
    ys = [y for _, y in coords.values()]
    pad = 0.3
    bounds = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

    xy = {b: bot_xy(trace, b, p, coords) for b, p in snapshot.bot_positions.items()}
    points = [(*xy[b][1], p["variance"]) for b, p in snapshot.bot_positions.items()]
    field = uncertainty_field(points, bounds)
    ax.imshow(field, extent=bounds, origin="lower", cmap="magma", alpha=0.55,
              vmin=0, vmax=max(1.0, field.max()), zorder=-1, aspect="auto")

    for u, v, attrs in trace.station.edges(data=True):
        style = EDGE_STYLE[edge_kind(attrs)]
        ax.plot([coords[u][0], coords[v][0]], [coords[u][1], coords[v][1]], zorder=1,
                color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"])

    for bot_id, path in trace.paths.items():
        for u, v in zip(path, path[1:]):
            ax.plot([coords[u][0], coords[v][0]], [coords[u][1], coords[v][1]],
                    color=bot_color(bot_id), linewidth=7, alpha=0.25, zorder=0)

    for node, (x, y) in coords.items():
        hazard = node in trace.hazards
        ax.scatter([x], [y], color="#5a1a1a" if hazard else "white", edgecolors="red" if hazard else BG,
                   s=220, zorder=2, linewidths=2.5 if hazard else 1.5)
        if hazard:
            ax.scatter([x], [y], color="red", marker="X", s=90, zorder=3)
        ax.annotate(node, (x, y), textcoords="offset points", xytext=(0, 14), fontsize=13,
                    fontweight="bold", color="#FF6B6B" if hazard else "white", ha="center", path_effects=OUTLINE)

    for bot_id, pos in snapshot.bot_positions.items():
        color = bot_color(bot_id)
        true_xy, believed_xy = xy[bot_id]
        ax.scatter([believed_xy[0]], [believed_xy[1]], color=color, s=90, marker="x", zorder=4, linewidths=2.5)
        ax.scatter([true_xy[0]], [true_xy[1]], color=color, s=140, marker="o", zorder=5,
                   edgecolors="white", linewidths=1.5)
        label = f"{bot_id} (waiting)" if pos["waiting"] else bot_id
        ax.annotate(label, true_xy, textcoords="offset points", xytext=(10, -12), fontsize=12,
                    color=color, fontweight="bold", path_effects=OUTLINE)

    handles = [Line2D([0], [0], **{k: v for k, v in s.items() if k != "label"}, label=s["label"])
               for s in EDGE_STYLE.values()]
    handles += [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="white",
               markersize=9, label="Bot: true position"),
        Line2D([0], [0], marker="x", color="white", markersize=9, linewidth=0, markeredgewidth=2.5,
               label="Bot: believed position (Kalman)"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="red", markeredgecolor="red",
               markersize=9, label="Hazard module"),
        Patch(facecolor="#B5367A", alpha=0.6, label="Position uncertainty (heatmap)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=11,
              ncol=1, frameon=False, labelcolor="white")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_timeline(trace, tick_idx):
    """Gantt: one row per accepted bot, one block per tick, plus airlock bands.

    x axis is in ticks (tick t occupies [t-1, t]). Ticks after the current
    one are dimmed; finished bots simply have no block. Airlock reservations
    come from schedule["reservations"] as (bot, start, end) per airlock.
    """
    bots = list(trace.paths)
    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.7 * len(bots)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    row = {b: i for i, b in enumerate(bots)}

    for key, slots in trace.schedule.get("reservations", {}).items():
        name = "-".join(key) if isinstance(key, tuple) else str(key)
        for bot, start, end in slots:
            if bot not in row:
                continue
            ax.barh(row[bot], end - start, left=start, height=0.9, color="none",
                    edgecolor="#8FA0FF", hatch="//", linewidth=1.2, zorder=1)
            ax.text((start + end) / 2, row[bot] + 0.5, name, ha="center", va="bottom",
                    fontsize=8, color="#8FA0FF", path_effects=OUTLINE)

    for snap in trace.ticks:
        for bot, pos in snap.bot_positions.items():
            kind = "waiting" if pos["state"] == "waiting" else (pos["thrust"] or "normal")
            ax.barh(row[bot], 1, left=snap.tick - 1, height=0.6, color=STATE_COLORS[kind],
                    edgecolor=BG, alpha=1.0 if snap.tick - 1 <= tick_idx else 0.25, zorder=2)

    ax.axvline(tick_idx + 1, color="white", linewidth=2, zorder=3)
    ax.set_yticks(range(len(bots)), bots, color="white")
    ax.set_ylim(len(bots) - 0.4, -0.7)
    ax.set_xlabel("tick", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in STATE_COLORS.items()]
              + [Patch(facecolor="none", edgecolor="#8FA0FF", hatch="//", label="airlock reservation")],
              loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, frameon=False, labelcolor="white")
    fig.tight_layout()
    return fig


def history_frames(trace, tick_idx):
    """(fuel_df, variance_df): one column per bot, indexed by tick, up to tick_idx.

    Bots that finished earlier are absent from later snapshots -> NaN.
    """
    rows = [(s.tick, {b: p for b, p in s.bot_positions.items()}) for s in trace.ticks[: tick_idx + 1]]
    index = [t for t, _ in rows]
    fuel = pd.DataFrame({b: [r.get(b, {}).get("fuel", np.nan) for _, r in rows] for b in trace.paths}, index=index)
    var = pd.DataFrame({b: [r.get(b, {}).get("variance", np.nan) for _, r in rows] for b in trace.paths}, index=index)
    return fuel, var


def airlock_edges(station):
    """Airlock edges as 'u-v' keys (sorted, matching the simulation's keys)."""
    return ["-".join(sorted((u, v))) for u, v, d in station.edges(data=True) if d.get("airlock")]


def command_sidebar():
    """Mission-control input: orders, hazards, cycling airlocks, then dispatch.

    Re-runs the whole pipeline (LLM parse -> Z3 verify -> route -> ticks) and
    resets the tick pointer. Priority/fuel stay at scenario defaults.
    """
    st.sidebar.header("🎛️ Mission control")
    scenarios = {}
    for bot_id, default in SCENARIOS.items():
        text = st.sidebar.text_area(f"{bot_id} command", value=default["text"], key=f"cmd_{bot_id}", height=80)
        scenarios[bot_id] = {**default, "text": text}

    base = run_simulation_base_station()
    hazards = st.sidebar.multiselect("Hazard modules", sorted(base.nodes), key="hazards")
    cycling = st.sidebar.multiselect("Cycling airlocks", airlock_edges(base), key="cycling")

    if st.sidebar.button("🚀 Dispatch", type="primary", width="stretch"):
        with st.spinner("Parsing commands, verifying, routing..."):
            st.session_state.trace = run_simulation(
                scenarios, set(hazards), {tuple(a.split("-")) for a in cycling})
        st.session_state.tick_idx = 0
        st.session_state.playing = False
    st.sidebar.caption("Freeform text → LLM function call → Z3 safety check → A* route.")


def run_simulation_base_station():
    """The station graph (from the cached trace if present) for sidebar options."""
    if "trace" not in st.session_state:
        st.session_state.trace = run_simulation()
    return st.session_state.trace.station


def tick_controls(n_ticks):
    """Prev / Play-Pause / Next buttons and the slider; mutates tick_idx."""
    ss = st.session_state
    with st.container(border=True):
        c_prev, c_play, c_next, c_label = st.columns([1, 1, 1, 5])
        if c_prev.button("⬅ Prev", disabled=ss.tick_idx <= 0, width="stretch"):
            ss.tick_idx -= 1
            ss.playing = False
        if c_play.button("⏸ Pause" if ss.playing else "▶ Play", width="stretch"):
            ss.playing = not ss.playing
            if ss.playing and ss.tick_idx >= n_ticks - 1:
                ss.tick_idx = 0
        if c_next.button("Next ➡", disabled=ss.tick_idx >= n_ticks - 1, width="stretch"):
            ss.tick_idx += 1
            ss.playing = False
        with c_label:
            st.markdown(f"<h4 style='text-align:center'>Tick {ss.tick_idx + 1} / {n_ticks}</h4>",
                        unsafe_allow_html=True)
            st.progress((ss.tick_idx + 1) / n_ticks)
        # The slider owns its key; write through it so programmatic changes stick.
        if n_ticks < 2:  # st.slider needs min < max
            return
        ss.slider_tick = ss.tick_idx
        ss.tick_idx = st.slider("Jump to tick", 0, n_ticks - 1, key="slider_tick", label_visibility="collapsed")


def kpi_row(trace, snap):
    """Tick, moving, waiting, fuel left, and airlock occupancy for this tick."""
    moving = sum(p["state"] == "moving" for p in snap.bot_positions.values())
    waiting = sum(p["state"] == "waiting" for p in snap.bot_positions.values())
    fuel = sum(p["fuel"] for p in snap.bot_positions.values())
    cols = st.columns(5)
    cols[0].metric("Tick", snap.tick)
    cols[1].metric("Moving", moving)
    cols[2].metric("Waiting", waiting)
    cols[3].metric("Fuel (active bots)", f"{fuel:.0f}")
    occ = "\n".join(f"{k}: {b or 'free'}" for k, b in snap.airlocks.items()) or "no airlocks"
    cols[4].markdown("**Airlock occupancy**")
    cols[4].code(occ, language=None)


def status_cards(trace, snap):
    """One card per active bot: fuel, variance, edge, positions, state, route info."""
    for bot_id, pos in snap.bot_positions.items():
        info = trace.route_info.get(bot_id, {})
        badge = "⏳ Waiting" if pos["state"] == "waiting" else f"🚀 Moving ({pos['thrust'] or 'normal'} thrust)"
        with st.container(border=True):
            st.markdown(f"<span style='color:{bot_color(bot_id)}; font-size:1.1em'><b>●</b></span> "
                        f"**{bot_id}** &nbsp; `{badge}`", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Fuel", f"{pos['fuel']:.0f}")
            m2.metric("Variance", f"{pos['variance']:.3f}")
            m3.metric("Route fuel", f"{info.get('fuel_cost', 0):.0f}")
            st.caption(f"Edge: {pos['edge'][0]} → {pos['edge'][1]}  ·  Airlocks on route: "
                       f"{', '.join(info.get('airlocks', [])) or 'none'}")
            st.caption(f"Believed: {pos['believed_pos']:.2f}  ·  True: {pos['true_pos']:.2f}")
    finished = [b for b in trace.paths if b not in snap.bot_positions]
    for b in finished:
        st.caption(f"🏁 {b} has finished its task.")


def events_panel(snap):
    """Events of the current tick only, grouped by bot, icon by event kind."""
    st.subheader("What the algorithm did this tick")
    if not snap.events:
        st.info("Nothing happened this tick.")
        return
    for bot in dict.fromkeys(e["bot"] for e in snap.events):
        with st.container(border=True):
            st.markdown(f"**{bot}**")
            for e in snap.events:
                if e["bot"] == bot:
                    st.write(f"{EVENT_ICONS.get(e['kind'], '•')} {e['text']}")


def tab_dispatch(trace):
    """Per-bot LLM attempts with Z3 verdicts; banner for rejected bots."""
    for bot_id, reason in trace.rejected.items():
        st.error(f"🚫 **{bot_id} rejected** — no safe request found. Z3: {reason}")
    for bot_id, attempts in trace.dispatch_log.items():
        with st.expander(f"{bot_id} — {len(attempts)} attempt(s)"):
            for a in attempts:
                v = a.get("verification", {})
                icon = "✅" if v.get("ok") else "❌"
                st.markdown(f"{icon} **Attempt {a.get('attempt')}** — request=`{a.get('request')}`  \n"
                            f"Z3 reason: _{v.get('reason')}_")


def tab_schedule(trace):
    """Airlock arbitration decisions and the reservation table."""
    decisions = trace.schedule.get("decisions", [])
    if not decisions:
        st.info("No conflict: no two bots contend for an airlock.")
    for d in decisions:
        st.success(f"**{d['winner']}** wins airlock **{d['airlock']}** over **{d['loser']}** — loser waits "
                   f"**{d['wait_ticks']}** tick(s). Reason: {d['reason']}")
    rows = [{"airlock": "-".join(k) if isinstance(k, tuple) else str(k), "bot": b, "start": s, "end": e}
            for k, slots in trace.schedule.get("reservations", {}).items() for b, s, e in slots]
    if rows:
        st.markdown("**Reservations**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def tab_pilot(trace):
    """Learned Q-values per (clutter/drift) state x action; best action marked."""
    rows = []
    for state, qs in trace.pilot_policy.items():
        best = max(qs, key=qs.get)
        rows.append({"state": state, **{a: round(q, 3) for a, q in qs.items()}, "best": f"★ {best}"})
    st.caption("Q-values learned in the pre-run training pass; the pilot picks the highest (epsilon=0.1 exploration).")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def tab_history(trace, tick_idx):
    """All events from tick 1 up to the current tick."""
    for s in trace.ticks[: tick_idx + 1]:
        if s.events:
            st.markdown(f"**Tick {s.tick}**")
            for e in s.events:
                st.write(f"{EVENT_ICONS.get(e['kind'], '•')} {e['text']}")


def main():
    """Page entry point: sidebar, controls, then every panel for the current tick."""
    command_sidebar()
    trace = st.session_state.trace
    ss = st.session_state
    ss.setdefault("tick_idx", 0)
    ss.setdefault("playing", False)

    st.title("🛰️ Orbital Cargo Bots — Mission Dashboard")
    st.caption("Pre-computed simulation trace; scrub step-by-step or press Play.")

    if not trace.ticks:
        st.warning("No bot was accepted, so there is nothing to simulate. See the verification log below.")
        tab_dispatch(trace)
        return

    n_ticks = len(trace.ticks)
    ss.tick_idx = min(ss.tick_idx, n_ticks - 1)
    tick_controls(n_ticks)
    snap = trace.ticks[ss.tick_idx]
    coords = node_layout(trace.station)

    kpi_row(trace, snap)

    map_col, status_col = st.columns([3, 2])
    with map_col:
        st.subheader("Station map")
        fig = draw_map(trace, snap, coords)
        st.pyplot(fig)
        plt.close(fig)
    with status_col:
        st.subheader("Bot status")
        status_cards(trace, snap)

    st.subheader("Timeline")
    fig = draw_timeline(trace, ss.tick_idx)
    st.pyplot(fig)
    plt.close(fig)

    fuel, var = history_frames(trace, ss.tick_idx)
    c1, c2 = st.columns(2)
    c1.subheader("Fuel per bot")
    c1.line_chart(fuel)
    c2.subheader("Kalman variance per bot")
    c2.line_chart(var)

    events_panel(snap)

    st.divider()
    st.subheader("Reasoning log")
    tabs = st.tabs(["📡 Dispatch & verification", "🚦 Airlock schedule", "🧠 Adaptive pilot", "📜 Event history"])
    with tabs[0]:
        tab_dispatch(trace)
    with tabs[1]:
        tab_schedule(trace)
    with tabs[2]:
        tab_pilot(trace)
    with tabs[3]:
        tab_history(trace, ss.tick_idx)

    if ss.playing:
        if ss.tick_idx < n_ticks - 1:
            time.sleep(0.6)
            ss.tick_idx += 1
            st.rerun()
        else:
            ss.playing = False


if __name__ == "__main__":
    main()

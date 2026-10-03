# Project Proposal — Orbital Cargo Bots

Track: **B (own idea)**

## What is your system's story?

I'm building a small orbital cargo station run by two autonomous cargo
bots. Mission control — a human operator — gives the bots freeform natural
language commands, like *"Move the O2 canister to Module C"* or *"Get bot 2
back to the docking bay before the burn window."* An LLM sits between the
human and the bots: it turns each command into a clean structured task
(which bot, what cargo, from where, to where). Before that task is allowed
to reach the bots' navigation and control systems, it has to pass a
mathematical safety check. If the check fails, the LLM doesn't just give
up — it's told *why* the plan is unsafe and gets a chance to propose a
safer alternative (different destination, smaller cargo load, wait for a
resource to free up).

The station itself is modeled as a handful of connected modules (5-6),
linked by corridors — one of which is a single-lane airlock the two bots
have to share.

## What are the hard safety rules?

The math verifier will check three things before any cargo task is
dispatched, using an SMT solver (Z3) rather than hand-written if/else logic:

1. **Mass limit** — does the destination module have structural capacity
   left for this cargo, given what's already stored there?
2. **Airlock state** — is the single shared airlock currently mid-cycle? If
   so, no route may pass through it right now.
3. **Hazard zones** — is the origin or destination module currently flagged
   as unsafe (e.g. a thermal or radiation event), making it temporarily
   off-limits?

If the solver comes back UNSAT, it should report *which* constraint failed
so the LLM has something concrete to replan around, rather than a generic
rejection.

## What is noisy in your world?

There's no GPS in orbit. Each bot has to estimate its own position by
dead-reckoning — integrating imprecise IMU/wheel-style movement readings
over time, which drift further from the truth the longer it goes
uncorrected. When a bot docks at a module, it gets a fiducial marker
reading — a clean, low-noise fix on its true position. I plan to fuse the
drifting dead-reckoned estimate with these periodic clean fixes using a
Kalman filter, so the system always has both a believed position and a
sense of how uncertain that belief currently is. That uncertainty is what
the dashboard should visualize.

## Where does conflict happen?

The two bots share exactly one resource that can't be used by both at
once: a single-lane airlock corridor connecting two of the station's
modules. When both bots' planned routes need that airlock in overlapping
windows, the system needs a rule to decide who goes first and who waits —
something like a priority auction (higher-priority task wins, with a
tie-break such as remaining fuel) rather than letting both bots collide or
deadlock.

## What interface will you build?

A Streamlit dashboard, since terminal output won't cut it for showing
spatial and probabilistic information at a glance. It should show:

- A live map of the station graph, with the two bots' true positions and
  their *believed* positions both plotted, so drift and correction are
  visible.
- A probability/uncertainty heatmap derived from the Kalman filter's
  covariance, sized or colored by how confident the system currently is.
- Planned paths for each bot.
- A running log of the LLM's reasoning: what it was asked, what it
  proposed, whether the verifier accepted or rejected it and why, and how
  any airlock conflict was resolved.

## Why this scenario

I chose an orbital station over the default warehouse benchmark because it
naturally forces every required mechanic to matter for a real reason: there
genuinely is no GPS in space (motivates the noisy-sensor/Kalman piece),
airlocks are a real single-lane bottleneck (motivates conflict resolution),
and mass/hazard limits are the kind of hard constraint that should never be
left to an LLM's judgment alone (motivates the non-negotiable SMT verifier).

## A note on the verifier

Per the assignment brief, the math verifier is the one piece I'm treating
as non-negotiable — everything else (dashboard polish, Q-table
sophistication, scenario complexity) can be simplified if time runs short
during the build, but the SMT-based safety check will be built and tested
on its own, before the LLM is even wired in, so the rest of the system has
something trustworthy to sit on top of.

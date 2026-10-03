"""
Tabular Q-learning for choosing thrust mode ("normal" vs "cautious") near
cluttered corridors on an orbital station, trading time cost against
drift-event risk.

State: (corridor_clutter_level, drift_risk_bucket)
    corridor_clutter_level in {"clear", "cluttered"}
    drift_risk_bucket      in {"low", "medium", "high"}
Action: "normal" or "cautious"

Plain dict Q-table, no RL framework, no neural nets.

Used by simulation.py: it trains a table on `simulate_outcome` (synthetic
rewards, next state drawn independently at random) before the tick loop, then
picks thrust epsilon-greedily on cluttered corridors. Cautious = 0.5 speed,
normal = 1.0. The learned preference is only as meaningful as the toy reward.
"""
import random

CLUTTER_LEVELS = ["clear", "cluttered"]
DRIFT_BUCKETS = ["low", "medium", "high"]
ACTIONS = ["normal", "cautious"]

STATES = [(c, d) for c in CLUTTER_LEVELS for d in DRIFT_BUCKETS]


class QTable:
    """Nested-dict Q-table: state -> {action: value}."""

    def __init__(self, states, actions, alpha=0.1, gamma=0.9):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.q = {s: {a: 0.0 for a in actions} for s in states}

    def choose_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.choice(self.actions)
        action_values = self.q[state]
        return max(action_values, key=action_values.get)

    def update(self, state, action, reward, next_state):
        """Standard Q-learning update rule."""
        best_next = max(self.q[next_state].values())
        current = self.q[state][action]
        self.q[state][action] = current + self.alpha * (
            reward + self.gamma * best_next - current
        )

    def best_action(self, state):
        """Greedy action (highest Q-value) for `state`."""
        action_values = self.q[state]
        return max(action_values, key=action_values.get)


def simulate_outcome(state, action):
    """
    Toy reward model, illustrative only (not real physics):
    - "cautious" always costs a small time penalty each step.
    - "normal" is faster (no time penalty) but risks a costly drift event.
    - drift-event probability rises with clutter and drift risk, and is
      cut roughly in half by choosing "cautious".
    """
    clutter, drift_risk = state

    base_risk = {"low": 0.05, "medium": 0.15, "high": 0.35}[drift_risk]
    if clutter == "cluttered":
        base_risk *= 2.0

    if action == "cautious":
        time_cost = -1.0
        drift_prob = base_risk * 0.5
    else:  # normal
        time_cost = -0.2
        drift_prob = base_risk

    drift_penalty = -20.0 if random.random() < drift_prob else 0.0
    return time_cost + drift_penalty


if __name__ == "__main__":
    random.seed(0)
    qtable = QTable(STATES, ACTIONS)

    episodes = 2000
    for _ in range(episodes):
        state = random.choice(STATES)
        action = qtable.choose_action(state, epsilon=0.2)
        reward = simulate_outcome(state, action)
        next_state = random.choice(STATES)  # states are drawn independently here
        qtable.update(state, action, reward, next_state)

    print("Learned Q-table and best action per state:")
    for state in STATES:
        values = qtable.q[state]
        best = qtable.best_action(state)
        print(f"  {state}: {values} -> best = {best}")

    risky_state = ("cluttered", "high")
    safe_state = ("clear", "low")
    print(f"\nBest action for {risky_state}: {qtable.best_action(risky_state)}")
    print(f"Best action for {safe_state}: {qtable.best_action(safe_state)}")

    assert qtable.best_action(risky_state) == "cautious", (
        "Expected cautious thrust to win in the high-risk cluttered state"
    )
    print("\nSelf-test passed: cautious thrust preferred in high-risk cluttered state.")

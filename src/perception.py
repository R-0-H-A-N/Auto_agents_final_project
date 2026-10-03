"""Noisy position tracking + 1D Kalman fusion for a cargo bot.

No GPS in space: position is a 1D scalar (cumulative progress along the
current corridor/edge, in arbitrary distance units). Dead reckoning drifts
via IMU-style noise; periodic fiducial markers give a clean fix that snaps
uncertainty back down.

Used by simulation.py: one BotPerception per bot; predict() every moving tick,
update() on arriving at a node (a docking fiducial fix). Noise comes from
np.random, which simulation.py seeds for reproducible runs.
"""

import numpy as np

# Process/measurement noise tunables (calibration knobs, not physics).
DRIFT_STD = 0.3          # per-tick Gaussian noise added to dead-reckoned position
PROCESS_VAR = DRIFT_STD ** 2   # Kalman process noise (Q)
FIDUCIAL_STD = 0.05      # clean marker reading noise
FIDUCIAL_VAR = FIDUCIAL_STD ** 2  # Kalman measurement noise (R)


class BotPerception:
    """Tracks one bot's believed position via a 1D Kalman filter.

    State is a single scalar (position). Velocity is treated as a known
    control input passed to predict(), not part of the filter state --
    keeps this at dimension 1 instead of the usual 2D [pos, vel] KF.
    """

    def __init__(self, initial_position: float = 0.0, initial_variance: float = 1.0):
        self.position = float(initial_position)   # believed position
        self.variance = float(initial_variance)    # uncertainty (P)

    def predict(self, velocity: float, dt: float = 1.0):
        """Advance belief by dead-reckoning: position += velocity*dt + drift
        noise, and grow variance by PROCESS_VAR. Returns (position, variance)."""
        drift = np.random.normal(0.0, DRIFT_STD)
        self.position += velocity * dt + drift
        self.variance += PROCESS_VAR  # uncertainty grows every uncorrected tick
        return self.position, self.variance

    def update(self, fiducial_position: float):
        """Fuse in a clean fiducial-marker reading (standard 1D KF update);
        variance shrinks sharply. Returns (position, variance)."""
        kalman_gain = self.variance / (self.variance + FIDUCIAL_VAR)
        self.position += kalman_gain * (fiducial_position - self.position)
        self.variance *= (1 - kalman_gain)  # sharply shrinks uncertainty
        return self.position, self.variance


def _simulate_true_trajectory(n_ticks: int, velocity: float) -> np.ndarray:
    return np.arange(n_ticks + 1) * velocity


if __name__ == "__main__":
    np.random.seed(42)

    n_ticks = 25
    velocity = 1.0
    fix_every = 6

    true_traj = _simulate_true_trajectory(n_ticks, velocity)
    bot = BotPerception(initial_position=0.0, initial_variance=0.1)

    var_before_fix = None
    var_after_fix = None

    print(f"{'tick':>4} {'true':>8} {'believed':>10} {'variance':>10}")
    for tick in range(1, n_ticks + 1):
        bot.predict(velocity)
        true_pos = true_traj[tick]

        if tick % fix_every == 0:
            var_before_fix = bot.variance
            fiducial_reading = true_pos + np.random.normal(0.0, FIDUCIAL_STD)
            bot.update(fiducial_reading)
            var_after_fix = bot.variance

        print(f"{tick:>4} {true_pos:>8.3f} {bot.position:>10.3f} {bot.variance:>10.5f}")

    assert var_after_fix < var_before_fix, "uncertainty should drop sharply after a fiducial fix"
    print("\nOK: uncertainty after last fix "
          f"({var_after_fix:.5f}) < uncertainty right before it ({var_before_fix:.5f})")

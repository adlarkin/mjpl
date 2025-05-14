from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Trajectory data.

    `n` is the number of states in the trajectory.
    The trajectory duration (with final time t_f) is `dt` * n.
    """

    # The timestep between each position, velocity, and acceleration snapshot.
    dt: float
    # Initial configuration (at t = 0).
    q_init: np.ndarray
    # Position snapshots at increments of dt, ranging from t = [dt, t_f].
    positions: list[np.ndarray]
    # Velocity snapshots at increments of dt, ranging from t = [dt, t_f]
    velocities: list[np.ndarray]
    # Acceleration snapshots at increments of dt, ranging from t = [dt, t_f]
    accelerations: list[np.ndarray]


class TrajectoryGenerator(ABC):
    """Abstract base class for generating trajectories."""

    @abstractmethod
    def generate_trajectory(self, waypoints: list[np.ndarray]) -> Trajectory | None:
        """Generate a trajectory.

        Args:
            waypoints: The waypoints for the trajectory to follow.

        Returns:
            A trajectory that follows `waypoints`, or None if a trajectory cannot be
            generated.
        """
        pass

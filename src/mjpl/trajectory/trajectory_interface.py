from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..types import Path


@dataclass(frozen=True)
class Trajectory:
    """Trajectory data.

    `n` is the number of states in the trajectory.
    The trajectory duration (with final time t_f) is `dt` * n.
    """

    # TODO: add q_init and joints? (see Path in types.py)

    # The timestep between each position, velocity, and acceleration snapshot.
    dt: float
    # Position snapshots at increments of dt, ranging from t = [dt, t_f].
    positions: list[np.ndarray]
    # Velocity snapshots at increments of dt, ranging from t = [dt, t_f]
    velocities: list[np.ndarray]
    # Acceleration snapshots at increments of dt, ranging from t = [dt, t_f]
    accelerations: list[np.ndarray]


class TrajectoryGenerator(ABC):
    """Abstract base class for generating trajectories."""

    @abstractmethod
    def generate_trajectory(self, path: Path) -> Trajectory:
        """Generate a trajectory.

        Args:
            path: The path for the trajectory to follow.

        Returns:
            A trajectory that follows `path`.
        """
        pass

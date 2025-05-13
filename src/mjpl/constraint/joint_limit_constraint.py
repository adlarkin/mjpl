import mujoco
import numpy as np

from .constraint_interface import Constraint


class JointLimitConstraint(Constraint):
    """Constraint that enforces joint limits on a configuration."""

    def __init__(self, model: mujoco.MjModel) -> None:
        """Constructor.

        Args:
            model: MuJoCo model, which contains the joint limits.
        """
        self.lower = model.jnt_range[:, 0]
        self.upper = model.jnt_range[:, 1]

    def valid_config(self, q: np.ndarray) -> bool:
        return np.all((q >= self.lower) & (q <= self.upper))

    def apply(self, q_old: np.ndarray, q: np.ndarray) -> np.ndarray | None:
        return q if self.valid_config(q) else None

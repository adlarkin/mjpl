import mujoco
import numpy as np

from .constraint_interface import Constraint


class JointLimitConstraint(Constraint):
    """Constraint that enforces joint limits on a configuration."""

    def __init__(self, model: mujoco.MjModel) -> None:
        """Constructor.

        Args:
            model: MuJoCo model.
        """
        self.model = model

    def valid_config(self, q: np.ndarray) -> bool:
        return np.all(
            (q >= self.model.jnt_range[:, 0]) & (q <= self.model.jnt_range[:, 1])
        )

    def apply(self, q: np.ndarray) -> np.ndarray | None:
        return q if self.valid_config(q) else None

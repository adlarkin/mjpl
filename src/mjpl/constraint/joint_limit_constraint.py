import numpy as np

from .constraint_interface import Constraint


class JointLimitConstraint(Constraint):
    """Constraint that enforces joint limits on a configuration."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        """Constructor.

        Args:
            lower: Lower bound on joint limits.
            upper: Upper bound on joint limits.
        """
        self.lower = lower
        self.upper = upper

    def valid_config(self, q: np.ndarray) -> bool:
        return np.all((q >= self.lower) & (q <= self.upper))

    def apply(self, q: np.ndarray) -> np.ndarray | None:
        return q if self.valid_config(q) else None

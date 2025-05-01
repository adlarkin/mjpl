from abc import ABC, abstractmethod

import numpy as np


class Constraint(ABC):
    """Abstract base class for a constraint."""

    @abstractmethod
    def valid_config(self, q: np.ndarray) -> bool:
        """Check if a configuration obeys a constraint.

        Args:
            q: The configuration.

        Returns:
            True if `q` obeys the constraint. False otherwise.
        """
        pass

    @abstractmethod
    def apply(self, q: np.ndarray) -> np.ndarray | None:
        """Apply a constraint to a configuration.

        Args:
            q: The configuration.

        Returns:
            A configuration derived from `q` that obeys the constraint, or None if
            deriving a configuration that obeys the constraint is not possible.
        """
        pass

from abc import ABC, abstractmethod

import numpy as np
from mink import SE3


class IKSolver(ABC):
    """Abstract base class for an inverse kinematics solver."""

    @abstractmethod
    def solve_ik(
        self,
        pose: SE3,
        site: str,
        q_init_guess: np.ndarray | None,
    ) -> np.ndarray | None:
        """Solve IK.

        Args:
            pose: The target pose, in the world frame.
            site: Name of the site for the target pose (i.e., the target frame).
            q_init_guess: Initial guess for the joint configuration.

        Returns:
            The joint configuration that satisfies the target pose within the
            allowed tolerances, or None if IK was unable to be solved.
        """
        pass

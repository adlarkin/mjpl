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
    ) -> list[np.ndarray]:
        """Solve IK.

        Args:
            pose: The target pose, in the world frame.
            site: Name of the site for the target pose (i.e., the target frame).
            q_init_guess: Initial guess for the joint configuration.

        Returns:
            A list of joint configurations that satisfy the target pose, or an empty
            list if IK was unable to be solved.
        """
        pass

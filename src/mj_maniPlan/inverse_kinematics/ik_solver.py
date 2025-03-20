from abc import ABC, abstractmethod

import numpy as np
from mink.lie.se3 import SE3


class IKSolver(ABC):
    """Abstract base class for an inverse kinematics solver."""

    def solve_ik(
        self,
        pose: SE3,
        site: str,
        pos_tolerance: float = 1e-3,
        ori_tolerance: float = 1e-3,
    ) -> np.ndarray | None:
        """Solve IK.

        Args:
            pose: The target pose.
            site: Name of the site for the target pose (i.e., the target frame).
            pos_tolerance: Allowed position error.
            ori_tolerance: Allowed orientation error.

        Returns:
            The joint configuration that satisfies the target pose within the
            allowed tolerances, or None if IK was unable to be solved.
        """
        return self._solve_ik_impl(pose, site, pos_tolerance, ori_tolerance)

    @abstractmethod
    def _solve_ik_impl(
        self, pose: SE3, site: str, pos_tolerance: float, ori_tolerance: float
    ) -> np.ndarray | None:
        pass

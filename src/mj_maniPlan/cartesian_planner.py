import numpy as np
from mink.lie.se3 import SE3

from .inverse_kinematics.ik_solver import IKSolver


def cartesian_plan(
    q_init_world: np.ndarray,
    poses: list[SE3],
    site: str,
    solver: IKSolver,
) -> list[np.ndarray]:
    """Plan joint configurations that satisfy a cartesian path.

    Args:
        q_init_world: Initial joint configuration of the world.
        poses: The cartesian path. These poses should be in the world frame.
        site: The site (i.e., frame) that should follow the cartesian path.
        solver: Solver used to compute IK for `poses` and `site`.

    Returns:
        A list of joint configurations, starting at `q_init_world`, that
        satisfies each pose in `poses`. If a configuration cannot be found for
        a pose, an empty list is returned.
    """
    path = [q_init_world]
    for p in poses:
        q = solver.solve_ik(p, site, path[-1])
        if q is None:
            print(f"Unable to find a joint configuration for pose {p}")
            return []
        path.append(q)
    return path

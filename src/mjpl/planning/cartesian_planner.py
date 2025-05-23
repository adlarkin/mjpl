import numpy as np
from mink.lie import SE3

from ..constraint.collision_constraint import CollisionConstraint
from ..constraint.constraint_interface import Constraint
from ..constraint.utils import obeys_constraints
from ..inverse_kinematics.ik_solver_interface import IKSolver
from .utils import _valid_collision_interval


def _interpolate_poses(
    pose_from: SE3, pose_to: SE3, lin_threshold: float, ori_threshold: float
) -> list[SE3]:
    """Interpolate two poses via decoupled translation and rotation.

    Args:
        pose_from: The interpolation start pose.
        pose_to: The interpolation end pose.
        lin_threshold: Maximum linear distance (in meters) allowed between adjacent poses.
        ori_threshold: Maximum orientation distance (in radians) allowed between adjacent poses.

    Returns:
        A list of poses starting at `pose_from` and ending at `pose_to` that are
        no further than `lin_threshold`/`ori_threshold` apart.
    """
    if lin_threshold <= 0.0:
        raise ValueError("`lin_threshold` must be > 0.0")
    if ori_threshold <= 0.0:
        raise ValueError("`ori_threshold` must be > 0.0")

    pose_diff = pose_to.minus(pose_from)
    lin_dist = np.linalg.norm(pose_diff[:3])
    ori_dist = np.linalg.norm(pose_diff[3:])

    lin_steps = int(np.ceil(lin_dist / lin_threshold))
    ori_steps = int(np.ceil(ori_dist / ori_threshold))
    num_steps = max(lin_steps, ori_steps, 1)

    return [
        pose_from.interpolate(pose_to, alpha)
        for alpha in np.linspace(0, 1, num_steps + 1)
    ]


def cartesian_plan(
    q_init: np.ndarray,
    poses: list[SE3],
    site: str,
    solver: IKSolver,
    constraints: list[Constraint],
    collision_interval_check: tuple[float, CollisionConstraint] | None = None,
    lin_threshold: float = 0.01,
    ori_threshold: float = 0.1,
) -> list[np.ndarray]:
    """Plan joint configurations that satisfy a Cartesian path.

    Args:
        q_init: Initial joint configuration.
        poses: The Cartesian path. These poses should be in the world frame.
        site: The site (i.e., frame) that should follow the Cartesian path.
        solver: Solver used to compute IK for `poses` and `site`.
        constraints: The constraints configurations in the Cartesian path must obey.
        collision_interval_check: A tuple that defines the step distance and
            CollisionConstraint that are used to check if the interval between two
            configurations obeys the CollisionConstraint. Interval checking is disabled
            if this is None.
        lin_threshold: The maximum linear distance (in meters) allowed between
            adjacent poses in `poses`. Pose interpolation will occur if this threshold
            is exceeded.
        ori_threshold: The maximum orientation distance (in radians) allowed between
            adjacent poses in `poses`. Pose interpolation will occur if this threshold
            is exceeded.

    Returns:
        A list of waypoints that adhere to a Cartesian path defined by `poses`,
        starting from `q_init`. If a path cannot be found, an empty list is returned.
    """
    if not site:
        raise ValueError("`site` must be defined.")

    interpolated_poses = [poses[0]]
    for i in range(len(poses) - 1):
        batch = _interpolate_poses(poses[i], poses[i + 1], lin_threshold, ori_threshold)
        interpolated_poses.extend(batch[1:])

    waypoints = [q_init]
    for p in interpolated_poses:
        configs = [
            q
            for q in solver.solve_ik(p, site, q_init_guess=waypoints[-1])
            if obeys_constraints(q, constraints)
            and (
                not collision_interval_check
                # Include endpoints in interval collision checks since:
                # 1. Interpolated poses may have been added
                # 2. It is not guaranteed that IK solutions account for constraints
                or _valid_collision_interval(
                    waypoints[-1], q, *collision_interval_check, include_endpoints=True
                )
            )
        ]
        if not configs:
            return []
        # Use the configuration that's closest to the previous waypoint.
        waypoints.append(min(configs, key=lambda q: np.linalg.norm(q - waypoints[-1])))
    return waypoints

import mujoco
import numpy as np
from mink.lie import SE3

from ..inverse_kinematics.ik_solver import IKSolver
from ..types import Path
from ..utils import all_joints


def _interpolate_poses(
    pose_from: SE3, pose_to: SE3, lin_threshold: float, ori_threshold: float
) -> list[SE3]:
    """Interpolate two poses via decoupled translation and rotation.

    Args:
        pose_from: The interpolation start pose.
        pose_to: The interpolation end pose.
        lin_threshold: Maximum linear distance allowed between adjacent poses.
        ori_threshold: Maximum orientation distance allowed between adjacent poses.

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

    poses = []
    for alpha in np.linspace(0, 1, num_steps + 1):
        poses.append(pose_from.interpolate(pose_to, alpha))
    return poses


def cartesian_plan(
    model: mujoco.MjModel,
    q_init_world: np.ndarray,
    poses: list[SE3],
    site: str,
    solver: IKSolver,
    lin_threshold: float = 0.01,
    ori_threshold: float = 0.1,
) -> Path | None:
    """Plan joint configurations that satisfy a Cartesian path.

    Args:
        model: MuJoCo model.
        q_init_world: Initial joint configuration of the world.
        poses: The Cartesian path. These poses should be in the world frame.
        site: The site (i.e., frame) that should follow the Cartesian path.
        solver: Solver used to compute IK for `poses` and `site`.
        lin_threshold: The maximum linear distance allowed between adjacent
            poses in `poses`. Pose interpolation will occur if this threshold
            is exceeded.
        ori_threshold: The maximum orientation distance allowed between adjacent
            poses in `poses`. Pose interpolation will occur if this threshold
            is exceeded.

    Returns:
        A path that adheres to a Cartesian path defined by `poses`, starting from
        `q_init_world`. None is returned if a path cannot be formed.
    """
    interpolated_poses = [poses[0]]
    for i in range(0, len(poses) - 1):
        batch = _interpolate_poses(poses[i], poses[i + 1], lin_threshold, ori_threshold)
        interpolated_poses.extend(batch[1:])

    waypoints = [q_init_world]
    for p in interpolated_poses:
        q = solver.solve_ik(p, site, waypoints[-1])
        if q is None:
            print(f"Unable to find a joint configuration for pose {p}")
            return None
        waypoints.append(q)
    return Path(q_init=q_init_world, waypoints=waypoints, joints=all_joints(model))

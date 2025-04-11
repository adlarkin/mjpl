import mujoco
import numpy as np

from ..collision_ruleset import CollisionRuleset
from ..joint_group import JointGroup
from .trajectory_interface import Trajectory, TrajectoryGenerator


def generate_trajectory(
    path: list[np.ndarray],
    generator: TrajectoryGenerator,
    q_init: np.ndarray,
    jg: JointGroup,
    cr: CollisionRuleset,
) -> Trajectory:
    """Generate a trajectory that follows a path and obeys a collision ruleset.

    This assumes that straight-line connections between adjacent waypoints in the path
    obey the collision ruleset. The following steps are taken to ensure the trajectory
    obeys the collision ruleset:
        1. Generate a trajectory.
        2. If part of the trajectory violates the collision ruleset, add an
           intermediate waypoint to the path segment that corresponds to the part of
           the trajectory that violates the collision ruleset.
        3. Repeat steps 1-2 until the trajectory has no segments that violate the
           collision ruleset.

    This is taken from section 3.5 of https://groups.csail.mit.edu/rrg/papers/Richter_ISRR13.pdf

    Args:
        path: The path the trajectory must follow.
        generator: Trajectory generator.
        q_init: The initial joint configuration of the world.
        jg: The joints used for the path and trajectory.
        cr: The collision ruleset the trajectory must adhere to.

    Returns:
        A trajectory that follows `path` without violating `cr`.
    """
    data = mujoco.MjData(jg.model)
    data.qpos = q_init
    filled_path = path.copy()
    while True:
        traj = generator.generate_trajectory(filled_path)
        for i in range(len(traj.positions)):
            q = traj.positions[i]
            jg.fk(q, data)
            mujoco.mj_collision(jg.model, data)
            if not cr.obeys_ruleset(data.contact.geom):
                # Add an intermediate waypoint to the section of the path
                # that corresponds to the trajectory position that's in collision.
                path_timestamps = _path_timing(filled_path, traj)
                collision_timestamp = (i + 1) * traj.dt
                _add_intermediate_waypoint(
                    filled_path, path_timestamps, collision_timestamp
                )
                break
        else:
            return traj


def _path_timing(path: list[np.ndarray], trajectory: Trajectory) -> list[float]:
    """Assign timestamps to waypoints in a path that correspond to a trajectory.

    Args:
        path: The path.
        trajectory: The trajectory that follows `path`.

    Returns:
        A list of timestamps for each waypoint in `path` based on `trajectory`.
    """
    # The first waypoint in the path corresponds to time 0.0
    timestamps = [0.0]
    if len(path) > 2:
        positions_array = np.stack(trajectory.positions)
        for waypoint in path[1:-1]:
            dists_sq = np.sum((positions_array - waypoint) ** 2, axis=1)
            timestamps.append((np.argmin(dists_sq) + 1) * trajectory.dt)
    # The last waypoint in the path corresponds to the trajectory duration
    timestamps.append(len(trajectory.positions) * trajectory.dt)

    return timestamps


def _add_intermediate_waypoint(
    path: list[np.ndarray], path_timing: list[float], timestamp: float
) -> None:
    """Insert an intermediate waypoint into a path segment that contains a timestamp.

    Args:
        path: The path that will have an intermediate waypoint added to it.
        path_timing: Timing information for the waypoints in `path`.
        timestamp: The timestamp that defines the segment of `path` that needs to have
            an intermediate waypoint added. If no segments in `path` contain this
            timestamp, no waypoint is added to `path`.
    """
    if len(path) != len(path_timing):
        raise ValueError("`path` and `path_timing` must be the same length.")
    for i in range(len(path) - 1):
        if path_timing[i] <= timestamp <= path_timing[i + 1]:
            intermediate_waypoint = (path[i] + path[i + 1]) / 2
            path.insert(i + 1, intermediate_waypoint)
            return

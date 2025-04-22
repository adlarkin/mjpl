import mujoco
import numpy as np

from .. import utils
from ..collision_ruleset import CollisionRuleset
from ..types import Path
from .trajectory_interface import Trajectory, TrajectoryGenerator


def generate_collision_free_trajectory(
    model: mujoco.MjModel,
    path: Path,
    generator: TrajectoryGenerator,
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
        cr: The collision ruleset the trajectory must adhere to.

    Returns:
        A trajectory that follows `path` without violating `cr`.
    """
    q_idx = utils.qpos_idx(model, path.joints)

    data = mujoco.MjData(model)
    data.qpos = path.q_init
    filled_path = path
    while True:
        traj = generator.generate_trajectory(filled_path)
        for i in range(len(traj.positions)):
            q = traj.positions[i]
            data.qpos[q_idx] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_collision(model, data)
            if not cr.obeys_ruleset(data.contact.geom):
                # Add an intermediate waypoint to the section of the path
                # that corresponds to the trajectory position that's in collision.
                waypoints = [wp for wp in filled_path.waypoints]
                path_timestamps = _path_timing(waypoints, traj)
                collision_timestamp = (i + 1) * traj.dt
                _add_intermediate_waypoint(
                    waypoints, path_timestamps, collision_timestamp
                )
                filled_path = Path(
                    q_init=path.q_init, waypoints=waypoints, joints=path.joints
                )
                break
        else:
            return traj


def _path_timing(waypoints: list[np.ndarray], trajectory: Trajectory) -> list[float]:
    """Assign timestamps to waypoints in a path that correspond to a trajectory.

    Args:
        path: The path.
        trajectory: The trajectory that follows `path`.

    Returns:
        A list of timestamps that correspond to path.waypoints based on `trajectory`.
    """
    if not waypoints:
        return []

    # The first waypoint maps to time 0.0
    timestamps = [0.0]
    if len(waypoints) > 2:
        positions_array = np.stack(trajectory.positions)
        for wp in waypoints[1:-1]:
            dists_sq = np.sum((positions_array - wp) ** 2, axis=1)
            timestamps.append((np.argmin(dists_sq) + 1) * trajectory.dt)
    # The last waypoint maps to the trajectory duration
    timestamps.append(len(trajectory.positions) * trajectory.dt)

    return timestamps


def _add_intermediate_waypoint(
    waypoints: list[np.ndarray], path_timing: list[float], timestamp: float
) -> None:
    """Insert an intermediate waypoint into a segment that contains a timestamp.

    Args:
        waypoints: The waypoints that will have an intermediate waypoint added to it.
        path_timing: Timing information for the waypoints in `path`.
        timestamp: The timestamp that defines the segment of `path` that needs to have
            an intermediate waypoint added. If no segments in `path` contain this
            timestamp, no waypoint is added to `path`.
    """
    if len(waypoints) != len(path_timing):
        raise ValueError("`waypoints` and `path_timing` must be the same length.")
    for i in range(len(waypoints) - 1):
        if path_timing[i] <= timestamp <= path_timing[i + 1]:
            intermediate_waypoint = (waypoints[i] + waypoints[i + 1]) / 2
            waypoints.insert(i + 1, intermediate_waypoint)
            return

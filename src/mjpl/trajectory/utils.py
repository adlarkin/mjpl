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
        model: MuJoCo model.
        path: The path the trajectory must follow.
        generator: Trajectory generator.
        cr: The collision ruleset the trajectory must adhere to.

    Returns:
        A trajectory that follows `path` without violating `cr`.
    """
    data = mujoco.MjData(model)
    while True:
        traj = generator.generate_trajectory(path)
        data.qpos = traj.q_init
        q_idx = utils.qpos_idx(model, traj.joints)
        for i in range(len(traj.positions)):
            q = traj.positions[i]
            data.qpos[q_idx] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_collision(model, data)
            if not cr.obeys_ruleset(data.contact.geom):
                # Add an intermediate waypoint to the section of the path
                # that corresponds to the trajectory position that's in collision.
                path_timestamps = _waypoint_timing(path.waypoints, traj)
                collision_timestamp = (i + 1) * traj.dt
                _add_intermediate_waypoint(
                    path.waypoints, path_timestamps, collision_timestamp
                )
                break
        else:
            return traj


def _waypoint_timing(
    waypoints: list[np.ndarray], trajectory: Trajectory
) -> list[float]:
    """Assign timestamps to waypoints that correspond to a trajectory.

    Args:
        waypoints: The waypoints.
        trajectory: The trajectory that follows `waypoints`.

    Returns:
        A list of timestamps for each waypoint in `waypoints` based on `trajectory`.
    """
    if len(waypoints) < 2:
        raise ValueError("There must be at least two waypoints defined.")

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
    waypoints: list[np.ndarray], timing: list[float], timestamp: float
) -> None:
    """Insert an intermediate waypoint into a segment that contains a timestamp.

    Args:
        waypoints: The waypoints that will have an intermediate waypoint added to it.
        timing: Timing information for `waypoints`.
        timestamp: The timestamp that defines the segment of `waypoints` that needs to
            have an intermediate waypoint added. If no segments in `waypoints` contain
            this timestamp, no waypoint is added to `waypoints`.
    """
    if len(waypoints) != len(timing):
        raise ValueError("`waypoints` and `timing` must be the same length.")
    for i in range(len(waypoints) - 1):
        if timing[i] <= timestamp <= timing[i + 1]:
            intermediate_waypoint = (waypoints[i] + waypoints[i + 1]) / 2
            waypoints.insert(i + 1, intermediate_waypoint)
            return

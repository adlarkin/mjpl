import numpy as np

from ..constraint.constraint_interface import Constraint
from ..constraint.utils import apply_constraints, obeys_constraints
from .trajectory_interface import Trajectory, TrajectoryGenerator


def generate_constrained_trajectory(
    waypoints: list[np.ndarray],
    generator: TrajectoryGenerator,
    constraints: list[Constraint],
) -> Trajectory | None:
    """Generate a trajectory that follows waypoints and obeys constraints.

    This assumes that straight-line connections between adjacent waypoints obey the
    constraints. The following steps are taken to ensure the trajectory obeys the
    constraints:
        1. Generate a trajectory.
        2. If part of the trajectory violates the constraints, add an intermediate
           waypoint to the segment of existing waypoints that corresponds to the part
           of the trajectory that violates the constraints.
        3. Repeat steps 1-2 until the trajectory has no segments that violate the
           constraints.

    This is taken from section 3.5 of https://groups.csail.mit.edu/rrg/papers/Richter_ISRR13.pdf

    Args:
        waypoints: Waypoints the trajectory must follow.
        generator: Trajectory generator.
        constraints: The constraints the trajectory must obey.

    Returns:
        A trajectory that follows `waypoints` without violating `constraints`, or None
        if a trajectory cannot be generated.
    """
    while True:
        traj = generator.generate_trajectory(waypoints)
        if traj is None:
            return None
        for i in range(len(traj.positions)):
            if not obeys_constraints(traj.positions[i], constraints):
                # Add an intermediate waypoint to the section of the path that
                # corresponds to the trajectory position that violates the constraints.
                path_timestamps = _waypoint_timing(waypoints, traj)
                trajectory_timestamp = (i + 1) * traj.dt
                if _add_intermediate_waypoint(
                    waypoints, path_timestamps, trajectory_timestamp, constraints
                ):
                    break
                # Adding an intermediate waypoint failed. This is probably because the
                # intermediate waypoint cannot obey the constraints.
                return None
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
    waypoints: list[np.ndarray],
    timing: list[float],
    timestamp: float,
    constraints: list[Constraint] = [],
) -> bool:
    """Insert a constrained waypoint into a waypoint segment that contains a timestamp.

    Args:
        waypoints: The waypoints that will have an intermediate waypoint added to it.
        timing: Timing information for `waypoints`.
        timestamp: The timestamp that defines the segment of `waypoints` that needs to
            have an intermediate waypoint added. If no segments in `waypoints` contain
            this timestamp, no waypoint is added to `waypoints`.
        constraints: The constraints the intermediate waypoint must obey.

    Returns:
        True if a constrained waypoint was added to `waypoints`. False otherwise.
    """
    if len(waypoints) != len(timing):
        raise ValueError("`waypoints` and `timing` must be the same length.")

    for i in range(len(waypoints) - 1):
        if timing[i] <= timestamp <= timing[i + 1]:
            intermediate_waypoint = (waypoints[i] + waypoints[i + 1]) / 2
            constrained_waypoint = apply_constraints(
                intermediate_waypoint, intermediate_waypoint, constraints
            )
            if constrained_waypoint is None:
                return False
            waypoints.insert(i + 1, constrained_waypoint)
            return True

    return False

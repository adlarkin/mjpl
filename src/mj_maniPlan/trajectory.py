from dataclasses import dataclass

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from .joint_group import JointGroup


@dataclass
class TrajectoryLimits:
    """Trajectory limits."""

    # The JointGroup the trajectory limits apply to.
    jg: JointGroup
    # Limits for the joints in the JointGroup.
    min_velocity: np.ndarray
    max_velocity: np.ndarray
    min_acceleration: np.ndarray
    max_acceleration: np.ndarray
    jerk: np.ndarray


@dataclass
class Trajectory:
    """Trajectory data for a list of waypoints.

    Let n=the number of states in the trajectory.
    Then the trajectory duration is `dt` * n.
    """

    # The JointGroup the configurations, velocities, and accelerations apply to.
    jg: JointGroup
    # The timestep between each configuration, velocity, and acceleration snapshot.
    dt: float
    # Configuration snapshots at increments of dt, ranging from t = [dt, t_f].
    configurations: list[np.ndarray]
    # Velocity snapshots at increments of dt, ranging from t = [dt, t_f]
    velocities: list[np.ndarray]
    # Acceleration snapshots at increments of dt, ranging from t = [dt, t_f]
    accelerations: list[np.ndarray]


def generate_trajectory(
    path: list[np.ndarray], limits: TrajectoryLimits, dt: float
) -> Trajectory:
    """Generate a trajectory that follows a path.

    The trajectory assumes zero velocity and acceleration at the start
    and end of the path.

    Args:
        path: The path that the trajectory should follow.
        limits: Limits to enforce in the trajectory.
        dt: The timestep (in seconds) between each point in the trajectory.

    Returns:
        A trajectory that follows `path` while adhering to `limits` at
        increments of `dt`.
    """
    dof = path[0].size
    otg = Ruckig(dof, dt, len(path))
    inp = InputParameter(dof)
    out = OutputParameter(dof, len(path))

    inp.current_position = path[0]
    inp.current_velocity = np.zeros(dof)
    inp.current_acceleration = np.zeros(dof)

    # NOTE: using intermediate waypoints invokes Ruckig's cloud API, which slows down trajectory generation time.
    # Pre-processing the path by filtering out some of the waypoints will make trajectory generation faster.
    # For more info, see https://docs.ruckig.com/md_pages_2__intermediate__waypoints.html
    inp.intermediate_positions = path[1:-1]

    inp.target_position = path[-1]
    inp.target_velocity = np.zeros(dof)
    inp.target_acceleration = np.zeros(dof)

    inp.min_velocity = limits.min_velocity
    inp.max_velocity = limits.max_velocity
    inp.min_acceleration = limits.min_acceleration
    inp.max_acceleration = limits.max_acceleration
    inp.max_jerk = limits.jerk

    configs = []
    vels = []
    accels = []

    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
        configs.append(np.array(out.new_position))
        vels.append(np.array(out.new_velocity))
        accels.append(np.array(out.new_acceleration))
        out.pass_to_input(inp)
    if res != Result.Finished:
        raise ValueError(
            "Did not successfully complete trajectory generation (this should not happen!)"
        )

    return Trajectory(limits.jg, dt, configs, vels, accels)

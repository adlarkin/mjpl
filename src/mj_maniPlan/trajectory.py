"""
APIs for generating a trajectory from a path with Rucking: https://ruckig.com/
"""

from dataclasses import dataclass

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from .joint_group import JointGroup


@dataclass
class TrajectoryLimits:
    # The JointGroup the trajectory limits apply to.
    jg: JointGroup
    # Limits for the joints in the JointGroup.
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray


# Trajectory data for a list of waypoints.
# n = the number of configurations, velocities, and accelerations.
# The duration of the trajectory is dt * n.
#
# The first state of the trajectory is the initial state.
@dataclass
class Trajectory:
    # The JointGroup the configurations, velocities, and accelerations apply to.
    jg: JointGroup
    # The timestep between each configuration, velocity, and acceleration snpashot.
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

    # The values chosen here are arbitrary and for demonstration purposes only.
    inp.max_velocity = limits.velocity
    inp.max_acceleration = limits.acceleration
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

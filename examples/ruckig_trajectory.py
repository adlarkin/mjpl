"""
APIs for generating a trajectory from a path with Rucking: https://ruckig.com/
"""

from dataclasses import dataclass

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig


# Trajectory data for a list of waypoints.
# n = the number of configurations, velocities, and accelerations.
# The duration of the trajectory is dt * n.
#
# The first state of the trajectory is the initial state.
@dataclass
class Trajectory:
    # The timestep between each configuration, velocity, and acceleration.
    dt: float
    # Configuration snapshots at increments of dt, ranging from t = [t_0, t_f].
    # Each configuration snapshot size is mjData.qpos
    configurations: list[list[float]]
    # Velocity snapshots at increments of dt, ranging from t = [t_0, t_f]
    # Each velocity snapshot size is mjData.qvel
    velocities: list[list[float]]
    # Acceleration snapshots at increments of dt, ranging from t = [t_0, t_f]
    # Each acceleration size is mjData.qacc
    accelerations: list[list[float]]


def generate_trajectory(dof, ctrl_cycle_rate, path) -> Trajectory:
    otg = Ruckig(dof, ctrl_cycle_rate, len(path))
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
    inp.max_velocity = np.ones(dof) * 2 * np.pi
    inp.max_acceleration = np.ones(dof) * np.pi
    inp.max_jerk = np.ones(dof) * 2 * np.pi

    # Set the beginning of the trajectory to the initial state.
    configs = [inp.current_position.copy()]
    vels = [inp.current_velocity.copy()]
    accels = [inp.current_acceleration.copy()]

    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
        configs.append(out.new_position)
        vels.append(out.new_velocity)
        accels.append(out.new_acceleration)
        out.pass_to_input(inp)
    if res != Result.Finished:
        raise ValueError(
            "Did not successfully complete trajectory generation (this should not happen!)"
        )

    return Trajectory(ctrl_cycle_rate, configs, vels, accels)

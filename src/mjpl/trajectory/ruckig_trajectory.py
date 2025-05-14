import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from .trajectory_interface import Trajectory, TrajectoryGenerator


class RuckigTrajectoryGenerator(TrajectoryGenerator):
    """Ruckig implementation of TrajectoryGenerator."""

    def __init__(
        self,
        dt: float,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        max_jerk: np.ndarray,
        min_velocity: np.ndarray | None = None,
        min_acceleration: np.ndarray | None = None,
    ):
        """Constructor.

        Args:
            dt: Trajectory timestep.
            max_velocity: Maximum allowed velocity of each joint.
            max_acceleration: Maximum allowed acceleration of each joint.
            max_jerk: Maximum allowed jerk of each joint.
            min_velocity: Minimum allowed velocity of each joint. If this is
                not set, the negative of max_velocity will be used.
            min_acceleration: Minimum allowed acceleration of each joint. If
                this is not set, the negative of max_acceleration will be used.
        """
        self.dt = dt
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity or -max_velocity
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration or -max_acceleration
        self.max_jerk = max_jerk

    def generate_trajectory(self, waypoints: list[np.ndarray]) -> Trajectory | None:
        dof = waypoints[0].size
        otg = Ruckig(dof, self.dt, len(waypoints))
        inp = InputParameter(dof)
        out = OutputParameter(dof, len(waypoints))

        inp.current_position = waypoints[0]
        inp.current_velocity = np.zeros(dof)
        inp.current_acceleration = np.zeros(dof)

        # NOTE: If Ruckig community version is installed, using intermediate
        # waypoints invokes Ruckig's cloud API, which slows down trajectory
        # generation time. Pre-processing the path by filtering out some of
        # the waypoints will make trajectory generation faster. For more info:
        # https://docs.ruckig.com/md_pages_2__intermediate__waypoints.html
        inp.intermediate_positions = waypoints[1:-1]

        inp.target_position = waypoints[-1]
        inp.target_velocity = np.zeros(dof)
        inp.target_acceleration = np.zeros(dof)

        inp.max_velocity = self.max_velocity
        inp.min_velocity = self.min_velocity
        inp.max_acceleration = self.max_acceleration
        inp.min_acceleration = self.min_acceleration
        inp.max_jerk = self.max_jerk

        positions = []
        velocities = []
        accelerations = []

        res = Result.Working
        while res == Result.Working:
            res = otg.update(inp, out)
            positions.append(np.array(out.new_position))
            velocities.append(np.array(out.new_velocity))
            accelerations.append(np.array(out.new_acceleration))
            out.pass_to_input(inp)
        if res != Result.Finished:
            return None

        return Trajectory(self.dt, waypoints[0], positions, velocities, accelerations)

import numpy as np
import toppra as ta

from .trajectory_interface import Trajectory, TrajectoryGenerator


class ToppraTrajectoryGenerator(TrajectoryGenerator):
    """TOPP-RA implementation of TrajectoryGenerator."""

    def __init__(
        self,
        dt: float,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        min_velocity: np.ndarray | None = None,
        min_acceleration: np.ndarray | None = None,
    ):
        """Constructor.

        Args:
            dt: Trajectory timestep.
            max_velocity: Maximum allowed velocity of each joint.
            max_acceleration: Maximum allowed acceleration of each joint.
            min_velocity: Minimum allowed velocity of each joint. If this is
                not set, the negative of max_velocity will be used.
            min_acceleration: Minimum allowed acceleration of each joint. If
                this is not set, the negative of max_acceleration will be used.
        """
        min_velocity = min_velocity or -max_velocity
        min_acceleration = min_acceleration or -max_acceleration

        velocity_limits = np.stack((min_velocity, max_velocity)).T
        acceleration_limits = np.stack((min_acceleration, max_acceleration)).T

        self.dt = dt
        self.velocity_constraint = ta.constraint.JointVelocityConstraint(
            velocity_limits
        )
        self.acceleration_constraint = ta.constraint.JointAccelerationConstraint(
            acceleration_limits
        )

    def generate_trajectory(self, waypoints: list[np.ndarray]) -> Trajectory | None:
        instance = ta.algorithm.TOPPRA(
            constraint_list=[self.velocity_constraint, self.acceleration_constraint],
            path=ta.SplineInterpolator(np.linspace(0, 1, len(waypoints)), waypoints),
            parametrizer="ParametrizeConstAccel",
        )
        trajectory = instance.compute_trajectory()
        if trajectory is None:
            return None
        t = np.arange(self.dt, trajectory.duration, self.dt)
        if not np.isclose(t[-1], trajectory.duration, rtol=0.0, atol=1e-8):
            t = np.append(t, trajectory.duration)
        return Trajectory(
            self.dt,
            waypoints[0],
            [position for position in trajectory(t)],
            [velocity for velocity in trajectory(t, order=1)],
            [acceleration for acceleration in trajectory(t, order=2)],
        )

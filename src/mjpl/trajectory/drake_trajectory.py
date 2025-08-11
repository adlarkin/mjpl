import numpy as np
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import Solve

from .trajectory_interface import Trajectory, TrajectoryGenerator


class DrakeTrajectoryGenerator(TrajectoryGenerator):
    """Drake KinematicTrajectoryOptimization implementation of TrajectoryGenerator."""

    def __init__(
        self,
        dt: float,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        velocity_limits: tuple[np.ndarray, np.ndarray] | None = None,
        acceleration_limits: tuple[np.ndarray, np.ndarray] | None = None,
        jerk_limits: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.dt = dt
        self.joint_limits = joint_limits
        self.velocity_limits = velocity_limits
        self.acceleration_limits = acceleration_limits
        self.jerk_limits = jerk_limits

    def generate_trajectory(self, waypoints: list[np.ndarray]) -> Trajectory | None:
        dof = len(waypoints[0])

        # TODO: make num_constrol_points (second arg) a param? Or at least, the
        # scaling factor w.r.t. number of waypoints?
        trajopt = KinematicTrajectoryOptimization(dof, len(waypoints) * 4)

        # TODO: make these costs params? (including pathlength)
        trajopt.AddDurationCost(1.0)
        trajopt.AddPathEnergyCost(1.0)

        prog = trajopt.get_mutable_prog()

        """
        # start constraint (start at first waypoint with zero velocity)
        trajopt.AddPathPositionConstraint(waypoints[0], waypoints[0], 0)
        prog.AddQuadraticErrorCost(np.eye(dof), waypoints[0], trajopt.control_points()[:, 0])
        trajopt.AddPathVelocityConstraint(np.zeros((dof, 1)), np.zeros((dof, 1)), 0)

        # goal constraint (end at last waypoint with zero velocity)
        trajopt.AddPathPositionConstraint(waypoints[-1], waypoints[-1], 1)
        prog.AddQuadraticErrorCost(np.eye(dof), waypoints[-1], trajopt.control_points()[:, -1])
        trajopt.AddPathVelocityConstraint(np.zeros((dof, 1)), np.zeros((dof, 1)), 1)

        # constraint to ensure trajectory passes through `waypoints` (in order)
        for i in range(1, len(waypoints) - 1):
            trajopt.AddPathPositionConstraint(waypoints[i], waypoints[i], i / (len(waypoints) - 1))
        """

        # constraint to ensure the trajectory passes through `waypoints` in the correct order
        for i in range(len(waypoints)):
            trajopt.AddPathPositionConstraint(
                waypoints[i], waypoints[i], i / (len(waypoints) - 1)
            )

        # trajectory should start and end with zero velocity
        trajopt.AddPathVelocityConstraint(np.zeros((dof, 1)), np.zeros((dof, 1)), 0)
        trajopt.AddPathVelocityConstraint(np.zeros((dof, 1)), np.zeros((dof, 1)), 1)

        if self.joint_limits:
            trajopt.AddPositionBounds(self.joint_limits[0], self.joint_limits[1])
        if self.velocity_limits:
            trajopt.AddVelocityBounds(self.velocity_limits[0], self.velocity_limits[1])
        if self.acceleration_limits:
            trajopt.AddAccelerationBounds(
                self.acceleration_limits[0], self.acceleration_limits[1]
            )
        if self.jerk_limits:
            trajopt.AddJerkBounds(self.jerk_limits[0], self.jerk_limits[1])

        # TODO: set an initial guess as the linear interpolation of waypoints?

        result = Solve(prog)
        if not result.is_success():
            return None

        positions = trajopt.ReconstructTrajectory(result)
        velocities = positions.MakeDerivative()
        accelerations = velocities.MakeDerivative()

        t = np.arange(self.dt, positions.end_time(), self.dt)
        if not np.isclose(t[-1], positions.end_time(), rtol=0.0, atol=1e-8):
            t = np.append(t, positions.end_time())

        return Trajectory(
            self.dt,
            waypoints[0],
            [pos for pos in positions.vector_values(t).T],
            [vels for vels in velocities.vector_values(t).T],
            [accs for accs in accelerations.vector_values(t).T],
        )

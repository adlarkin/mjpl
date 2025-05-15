import mink
import mujoco
import numpy as np
from mink import SE3

from .. import utils
from ..constraint.constraint_interface import Constraint
from ..constraint.utils import obeys_constraints
from .ik_solver_interface import IKSolver


class MinkIKSolver(IKSolver):
    """Mink implementation of IKSolver."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joints: list[str],
        constraints: list[Constraint] = [],
        pos_tolerance: float = 1e-3,
        ori_tolerance: float = 1e-3,
        seed: int | None = None,
        max_attempts: int = 1,
        iterations: int = 500,
        qp_solver: str = "daqp",
    ):
        """Constructor.

        Args:
            model: MuJoCo model.
            joints: The joints that can be manipulated to solve IK. The values of these
                joints will be randomized when generating initial states for new solve
                attempts.
            constraints: The constraints to enforce on IK solutions.
            pos_tolerance: Allowed position error (meters).
            ori_tolerance: Allowed orientation error (radians).
            seed: Seed used for generating random samples in the case of retries
                (see `max_attempts`).
            max_attempts: Maximum number of solve attempts.
            iterations: Maximum iterations to run the solver for, per attempt.
            qp_solver: QP Solver to use, which comes from the qpsolvers package:
                https://github.com/qpsolvers/qpsolvers
        """
        if not joints:
            raise ValueError("`joints` cannot be empty.")
        if max_attempts < 1:
            raise ValueError("`max_attempts` must be > 0.")
        if iterations < 1:
            raise ValueError("`iterations` must be > 0.")
        self.model = model
        self.joints = joints
        self.constraints = constraints
        self.pos_tolerance = pos_tolerance
        self.ori_tolerance = ori_tolerance
        self.seed = seed
        self.max_attempts = max_attempts
        self.iterations = iterations
        self.qp_solver = qp_solver

        # If needed, create a damping task to make sure joints that are not in
        # self.joints are held fixed while solving IK.
        self.damping_task: mink.DampingTask | None = None
        all_joints = utils.all_joints(model)
        if set(joints) != set(all_joints):
            fixed_joints = [j for j in all_joints if j not in joints]
            cost = np.zeros((model.nv,))
            cost[utils.qvel_idx(model, fixed_joints)] = 1e9
            self.damping_task = mink.DampingTask(model, cost)

    def solve_ik(
        self, pose: SE3, site: str, q_init_guess: np.ndarray | None
    ) -> np.ndarray | None:
        end_effector_task = mink.FrameTask(
            frame_name=site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=0.1,
        )
        end_effector_task.set_target(pose)

        tasks = [end_effector_task]
        if self.damping_task is not None:
            tasks.append(self.damping_task)

        limits = [mink.ConfigurationLimit(self.model)]

        configuration = mink.Configuration(self.model)
        configuration.update(q_init_guess)

        for attempt in range(self.max_attempts):
            for _ in range(self.iterations):
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= self.pos_tolerance
                ori_achieved = np.linalg.norm(err[3:]) <= self.ori_tolerance
                if pos_achieved and ori_achieved:
                    if obeys_constraints(configuration.q, self.constraints):
                        return configuration.q
                    break
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    self.model.opt.timestep,
                    solver=self.qp_solver,
                    damping=1e-3,
                    limits=limits,
                )
                configuration.integrate_inplace(vel, self.model.opt.timestep)

            # Make sure a different seed is used for each randomly generated config.
            _seed = self.seed + attempt if self.seed is not None else self.seed
            next_guess = utils.random_config(
                self.model, configuration.q, self.joints, _seed, self.constraints
            )
            configuration.update(next_guess)
        return None

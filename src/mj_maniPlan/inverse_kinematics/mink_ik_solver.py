import mink
import mujoco
import numpy as np

from .. import utils
from ..collision_ruleset import CollisionRuleset
from ..joint_group import JointGroup
from .ik_solver import IKSolver


class MinkIKSolver(IKSolver):
    """Mink implementation of IKSolver."""

    def __init__(
        self,
        model: mujoco.MjModel,
        jg: JointGroup,
        cr: CollisionRuleset | None = None,
        q_init_guess: np.ndarray | None = None,
        seed: int | None = None,
        max_attempts: int = 1,
        iterations: int = 500,
        solver: str = "quadprog",
    ):
        """Constructor.

        Args:
            model: MuJoCo model.
            jg: The joints to use when generating initial states for new solve
                attempts and validating configurations.
            cr: The collision rules to enforce. If defined, IK solutions must
                also obey this ruleset.
            q_init_guess: Initial guess for the joint configuration.
            seed: Seed used for generating random samples in the case of retries
                (see `max_attempts`).
            max_attempts: Maximum number of solve attempts.
            iterations: Maximum iterations to run the solver for, per attempt.
            solver: Solver to use, which comes from the qpsolvers package:
                https://github.com/qpsolvers/qpsolvers
        """
        if max_attempts < 1:
            raise ValueError("`max_attempts` must be > 0.")
        if iterations < 1:
            raise ValueError("`iterations` must be > 0.")
        self.model = model
        self.data = mujoco.MjData(model)
        self.jg = jg
        self.cr = cr
        self.q_init_guess = q_init_guess
        self.rng = np.random.default_rng(seed=seed)
        self.max_attempts = max_attempts
        self.iterations = iterations
        self.solver = solver

    def _solve_ik_impl(
        self, pose: mink.lie.SE3, site: str, pos_tolerance: float, ori_tolerance: float
    ) -> np.ndarray | None:
        end_effector_task = mink.FrameTask(
            frame_name=site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        end_effector_task.set_target(pose)
        tasks = [end_effector_task]

        limits = [mink.ConfigurationLimit(self.model)]

        configuration = mink.Configuration(self.model)
        configuration.update(self.q_init_guess)

        for _ in range(self.max_attempts):
            for _ in range(self.iterations):
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_tolerance
                ori_achieved = np.linalg.norm(err[3:]) <= ori_tolerance
                if pos_achieved and ori_achieved:
                    if utils.is_valid_config(
                        configuration.q[self.jg.qpos_addrs], self.jg, self.data, self.cr
                    ):
                        return configuration.q
                    break
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    self.model.opt.timestep,
                    solver=self.solver,
                    damping=1e-3,
                    limits=limits,
                )
                configuration.integrate_inplace(vel, self.model.opt.timestep)

            next_guess = configuration.q
            next_guess[self.jg.qpos_addrs] = utils.random_valid_config(
                self.rng, self.jg, self.data, self.cr
            )
            configuration.update(next_guess)
        return None

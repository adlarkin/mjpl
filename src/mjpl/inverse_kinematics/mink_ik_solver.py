import mink
import mujoco
import numpy as np

from .. import utils
from ..collision_ruleset import CollisionRuleset
from .ik_solver import IKSolver


class MinkIKSolver(IKSolver):
    """Mink implementation of IKSolver."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joints: list[str],
        cr: CollisionRuleset | None = None,
        pos_tolerance: float = 1e-3,
        ori_tolerance: float = 1e-3,
        seed: int | None = None,
        max_attempts: int = 1,
        iterations: int = 500,
        solver: str = "daqp",
    ):
        """Constructor.

        Args:
            model: MuJoCo model.
            joints: The joints to use when generating initial states for new solve
                attempts and validating configurations.
            cr: The collision rules to enforce. If defined, IK solutions must
                also obey this ruleset.
            pos_tolerance: Allowed position error.
            ori_tolerance: Allowed orientation error.
            seed: Seed used for generating random samples in the case of retries
                (see `max_attempts`).
            max_attempts: Maximum number of solve attempts.
            iterations: Maximum iterations to run the solver for, per attempt.
            solver: Solver to use, which comes from the qpsolvers package:
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
        self.cr = cr
        self.pos_tolerance = pos_tolerance
        self.ori_tolerance = ori_tolerance
        self.seed = seed
        self.max_attempts = max_attempts
        self.iterations = iterations
        self.solver = solver

    def solve_ik(
        self, pose: mink.lie.SE3, site: str, q_init_guess: np.ndarray | None
    ) -> np.ndarray | None:
        data = mujoco.MjData(self.model)

        end_effector_task = mink.FrameTask(
            frame_name=site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=0.1,
        )
        end_effector_task.set_target(pose)
        tasks = [end_effector_task]

        limits = [mink.ConfigurationLimit(self.model)]

        configuration = mink.Configuration(self.model)
        configuration.update(q_init_guess)

        for attempt in range(self.max_attempts):
            for _ in range(self.iterations):
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= self.pos_tolerance
                ori_achieved = np.linalg.norm(err[3:]) <= self.ori_tolerance
                if pos_achieved and ori_achieved:
                    data.qpos = configuration.q
                    if utils.is_valid_config(self.model, data, self.cr):
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

            # Make sure a different seed is used for each randomly generated config.
            _seed = self.seed + attempt if self.seed is not None else self.seed
            next_guess = utils.random_valid_config(
                self.model, configuration.q, self.joints, _seed, self.cr
            )
            configuration.update(next_guess)
        return None

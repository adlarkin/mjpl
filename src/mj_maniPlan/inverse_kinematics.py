from dataclasses import dataclass

import mink
import mujoco
import mujoco.viewer
import numpy as np

from . import utils
from .collision_ruleset import CollisionRuleset
from .joint_group import JointGroup


@dataclass
class IKOptions:
    """Options for solving inverse kinematics."""

    # The joints to use when generating initial states for new solve attempts and validating configurations.
    jg: JointGroup = None
    # The collision rules to enforce. If `None`, this disables collision checks.
    cr: CollisionRuleset | None = None
    # Seed used for generating random samples in the case of retries.
    seed: int | None = None
    # Allowed position error.
    pos_tolerance: float = 1e-3
    # Allowed orientation error.
    ori_tolerance: float = 1e-3
    # Maximum iterations to run for the IK solver.
    iterations: int = 500
    # Solver to use. This comes from the qpsolvers package:
    # https://github.com/qpsolvers/qpsolvers
    solver: str = "quadprog"
    # Maximum number of solve attempts.
    max_attempts: int = 1


def solve_ik(
    site: str,
    q_init_guess: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    opts: IKOptions = IKOptions(),
) -> np.ndarray | None:
    """Solve inverse kinematics for a given pose.

    Args:
        site: Name of the site for the target pose (i.e., the target frame).
        q_init_guess: The initial guess for the joint configuration.
        target_pos: Desired position.
        target_rot: Desired orientation, expressed as a 3x3 rotation matrix.
        opts: Options for customizing IK solution behavior.

    Returns:
        A joint configuration that achieves a pose within the tolerance defined by
        `opts` for frame `site`. Returns None if a joint configuration is unable
        to be found.
    """
    model = opts.jg.model
    data = mujoco.MjData(model)
    limits = [mink.ConfigurationLimit(model)]
    rng = np.random.default_rng(seed=opts.seed)

    configuration = mink.Configuration(model)
    end_effector_task = mink.FrameTask(
        frame_name=site,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    end_effector_task.set_target(
        mink.lie.SE3.from_rotation_and_translation(
            mink.lie.SO3.from_matrix(target_rot),
            target_pos,
        )
    )
    tasks = [end_effector_task]

    for attempt_idx in range(opts.max_attempts):
        # Initialize the state for IK.
        q_init = q_init_guess.copy()
        if attempt_idx > 0:
            q_init[opts.jg.joint_ids] = utils.random_valid_config(
                rng,
                opts.jg,
                data,
                opts.cr,
            )
        configuration.update(q_init)

        # Attempt to solve IK.
        for _ in range(opts.iterations):
            err = end_effector_task.compute_error(configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= opts.pos_tolerance
            ori_achieved = np.linalg.norm(err[3:]) <= opts.ori_tolerance
            if pos_achieved and ori_achieved:
                is_collision_free = (opts.cr is not None) and utils.is_valid_config(
                    configuration.q[opts.jg.joint_ids], opts.jg, data, opts.cr
                )
                if not is_collision_free:
                    print(
                        f"IK solve attempt {attempt_idx + 1}/{opts.max_attempts} in collision."
                    )
                    break

                return configuration.q

            vel = mink.solve_ik(
                configuration,
                tasks,
                model.opt.timestep,
                solver=opts.solver,
                damping=1e-3,
                limits=limits,
            )
            configuration.integrate_inplace(vel, model.opt.timestep)

    return None

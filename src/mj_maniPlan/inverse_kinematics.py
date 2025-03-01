from dataclasses import dataclass

import mink
import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class IKOptions:
    """Options for solving inverse kinematics."""

    # Allowed position error.
    pos_tolerance: float = 1e-3
    # Allowed orientation error.
    ori_tolerance: float = 1e-3
    # Maximum iterations to run for the IK solver.
    iterations: int = 500
    # Configuration used as the initial state for the IK solver.
    q_init: np.ndarray | None = None
    # Solver to use. This comes from the qpsolvers package:
    # https://github.com/qpsolvers/qpsolvers
    solver: str = "quadprog"


def solve_ik(
    model: mujoco.MjModel,
    site: str,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    opts: IKOptions = IKOptions(),
) -> np.ndarray | None:
    """Solve inverse kinematics for a given pose.

    Args:
        model: MuJoCo model.
        site: Name of the site for the target pose (i.e., the target frame).
        target_pos: Desired position.
        target_rot: Desired orientation, expressed as a 3x3 rotation matrix.
        opts: Options for customizing IK solution behavior.

    Returns:
        A joint configuration that achieves a pose within the tolerance defined by
        `opts` for frame `site`. Returns None if a joint configuration is unable
        to be found.
    """
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

    limits = [mink.ConfigurationLimit(model)]

    configuration = mink.Configuration(model)
    configuration.update(opts.q_init)

    for _ in range(opts.iterations):
        err = end_effector_task.compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= opts.pos_tolerance
        ori_achieved = np.linalg.norm(err[3:]) <= opts.ori_tolerance
        if pos_achieved and ori_achieved:
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

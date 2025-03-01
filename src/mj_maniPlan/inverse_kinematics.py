from dataclasses import dataclass

import mink
import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class IKOptions:
    """Options for solving inverse kinematics."""

    pos_tolerance: float = 1e-3
    ori_tolerance: float = 1e-3
    iterations: int = 500
    q_init: np.ndarray | None = None
    solver: str = "quadprog"


def solve_ik(
    model: mujoco.MjModel,
    site: str,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    opts: IKOptions = IKOptions(),
) -> np.ndarray | None:
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

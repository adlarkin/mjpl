import mujoco
import numpy as np

from .configuration import Configuration


def configuration_distance(q_from: np.ndarray, q_to: np.ndarray):
    return np.linalg.norm(q_to - q_from)


# NOTE: this will modify `data` in-place.
def is_valid_config(
    q: np.ndarray,
    config: Configuration,
    data: mujoco.MjData,
) -> bool:
    # Check joint limits.
    if not ((q >= config.lower_limits) & (q <= config.upper_limits)).all():
        return False

    # Check for collisions.
    # We have to run FK once data.qpos is updated before running the collision checker.
    config.fk(q, data)
    mujoco.mj_collision(config.model, data)
    return not data.ncon


# TODO: debug this (write tests for it?), it looks like it's giving invalid configs at times.
# For example, the position_ctrl.py script sometimes gives 'q_goal is not a valid config' error
# NOTE: this will modify `data` in-place, since it calls is_valid_config internally.
def random_valid_config(
    rng: np.random.Generator,
    config: Configuration,
    data: mujoco.MjData,
) -> np.ndarray:
    q_rand = config.random_config(rng)
    while not is_valid_config(q_rand, config, data):
        q_rand = config.random_config(rng)
    return q_rand


def site_pose(site_name: str, data: mujoco.MjData):
    pos = data.site(site_name).xpos
    rot_mat = data.site(site_name).xmat.reshape(3, 3)
    return pos, rot_mat

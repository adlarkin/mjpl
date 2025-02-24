import mujoco
import numpy as np

from .collision_ruleset import CollisionRuleset
from .joint_group import JointGroup


def configuration_distance(q_from: np.ndarray, q_to: np.ndarray):
    return np.linalg.norm(q_to - q_from)


# NOTE: this will modify `data` in-place.
def is_valid_config(
    q: np.ndarray,
    jg: JointGroup,
    data: mujoco.MjData,
    cr: CollisionRuleset,
) -> bool:
    # Check joint limits.
    if not ((q >= jg.lower_limits) & (q <= jg.upper_limits)).all():
        return False

    # Check for collisions.
    # We have to run FK once data.qpos is updated before running the collision checker.
    jg.fk(q, data)
    mujoco.mj_collision(jg.model, data)
    return cr.obeys_ruleset(data)


# NOTE: this will modify `data` in-place, since it calls is_valid_config internally.
def random_valid_config(
    rng: np.random.Generator,
    jg: JointGroup,
    data: mujoco.MjData,
    cr: CollisionRuleset,
) -> np.ndarray:
    q_rand = jg.random_config(rng)
    while not is_valid_config(q_rand, jg, data, cr):
        q_rand = jg.random_config(rng)
    return q_rand


def site_pose(site_name: str, data: mujoco.MjData):
    pos = data.site(site_name).xpos
    rot_mat = data.site(site_name).xmat.reshape(3, 3)
    return pos, rot_mat

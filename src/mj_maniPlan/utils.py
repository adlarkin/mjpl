import numpy as np
import mujoco


def configuration_distance(q_from: np.ndarray, q_to: np.ndarray):
    return np.linalg.norm(q_to - q_from)

def joint_names_to_qpos_addrs(joint_names: list[str], model: mujoco.MjModel) -> np.ndarray:
    return np.array([ model.joint(name).qposadr.item() for name in joint_names ])

def joint_limits(joint_names: list[str], model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lower_limits = np.array([ model.joint(name).range[0] for name in joint_names ])
    upper_limits = np.array([ model.joint(name).range[1] for name in joint_names ])
    return lower_limits, upper_limits

def random_config(rng: np.random.Generator, lower_limits: np.ndarray, upper_limits: np.ndarray) -> np.ndarray:
    return rng.uniform(low=lower_limits, high=upper_limits)

def fk(q: np.ndarray, qpos_addrs: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData):
    data.qpos[qpos_addrs] = q
    mujoco.mj_kinematics(model, data)

# NOTE: this will modify `data` in-place.
def is_valid_config(
    q: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    qpos_addrs: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData
) -> bool:
    # Check joint limits.
    if not ((q >= lower_limits) & (q <= upper_limits)).all():
        return False

    # Check for collisions.
    # We have to run FK once data.qpos is updated before running the collision checker.
    fk(q, qpos_addrs, model, data)
    mujoco.mj_collision(model, data)
    return not data.ncon

# NOTE: this will modify `data` in-place, since it calls is_valid_config internally.
def random_valid_config(
    rng: np.random.Generator,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    joint_qpos_addrs: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData
) -> np.ndarray:
    q_rand = random_config(rng, lower_limits, upper_limits)
    while not is_valid_config(q_rand, lower_limits, upper_limits, joint_qpos_addrs, model, data):
        q_rand = random_config(rng, lower_limits, upper_limits)
    return q_rand

def site_pose(site_name: str, data: mujoco.MjData):
    pos = data.site(site_name).xpos
    rot_mat = data.site(site_name).xmat.reshape(3,3)
    return pos, rot_mat

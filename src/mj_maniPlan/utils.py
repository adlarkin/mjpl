import numpy as np
import mujoco


def configuration_distance(q_from, q_to):
    return np.linalg.norm(q_to - q_from)

def joint_names_to_qpos_addrs(joint_names, model: mujoco.MjModel):
    return np.array([ model.joint(name).qposadr.item() for name in joint_names ])

def joint_limits(joint_names, model: mujoco.MjModel):
    lower_limits = np.array([ model.joint(name).range[0] for name in joint_names ])
    upper_limits = np.array([ model.joint(name).range[1] for name in joint_names ])
    return lower_limits, upper_limits

def random_config(rng: np.random.Generator, lower_limits, upper_limits):
    return rng.uniform(low=lower_limits, high=upper_limits)

# NOTE: this will modify `data` in-place.
def is_valid_config(q, lower_limits, upper_limits, qpos_addrs, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    # Check joint limits.
    if not ((q >= lower_limits) & (q <= upper_limits)).all():
        return False

    # Check for collisions.
    # We have to run FK once data.qpos is updated before running the collision checker.
    # TODO: enforce padding on the collision check? Not sure how to do this in MuJoCo yet
    data.qpos[qpos_addrs] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    return not data.ncon

# NOTE: this will modify `data` in-place, since it calls is_valid_config internally.
def random_valid_config(rng: np.random.Generator, lower_limits, upper_limits, joint_qpos_addrs, model: mujoco.MjModel, data: mujoco.MjData):
    q_rand = random_config(rng, lower_limits, upper_limits)
    while not is_valid_config(q_rand, lower_limits, upper_limits, joint_qpos_addrs, model, data):
        q_rand = random_config(rng, lower_limits, upper_limits)
    return q_rand

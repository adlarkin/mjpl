import mujoco
import numpy as np


# Get the joint addresses in the model's corresponding data.qpos for the named joints.
# This assumes that `joint_names` refer to joints with 1 DOF.
def joint_names_to_qpos_addrs(
    joint_names: list[str], model: mujoco.MjModel
) -> np.ndarray:
    return np.array([model.joint(name).qposadr.item() for name in joint_names])


# Get the joint addresses in the model's corresponding data.qvel/data.qacc for the named joints.
# This assumes that `joint_names` refer to joints with 1 DOF.
def joint_names_to_dof_addrs(
    joint_names: list[str], model: mujoco.MjModel
) -> np.ndarray:
    return np.array([model.joint(name).dofadr.item() for name in joint_names])


def joint_limits(
    joint_names: list[str], model: mujoco.MjModel
) -> tuple[np.ndarray, np.ndarray]:
    lower_limits = np.array([model.joint(name).range[0] for name in joint_names])
    upper_limits = np.array([model.joint(name).range[1] for name in joint_names])
    return lower_limits, upper_limits


# Wrapper for handling a joint configuration.
# This is useful when a user is interested in interacting with a subset of the joints in MjData.
# This class assumes the joint(s) of interest have 1 DOF.
class Configuration:
    def __init__(self, joint_names: list[str], model: mujoco.MjModel):
        self.joint_names = joint_names
        self.model = model
        self.qpos_addrs = joint_names_to_qpos_addrs(joint_names, model)
        self.qvel_qacc_addrs = joint_names_to_dof_addrs(joint_names, model)
        self.lower_limits, self.upper_limits = joint_limits(joint_names, model)

    # Create a random joint configuration within the joint limits
    def random_config(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(low=self.lower_limits, high=self.upper_limits)

    # Run forward kinematics on the specified joint configuration.
    # The order of the values in q should match the order of the configuration's joint names.
    def fk(self, q: np.ndarray, data: mujoco.MjData) -> None:
        assert q.size == self.qpos_addrs.size
        data.qpos[self.qpos_addrs] = q
        mujoco.mj_kinematics(self.model, data)
        # TODO: call to mj_comPos for updated Jacobians?
        # https://github.com/kevinzakka/mink/blob/cf1a302ff31b1f620abecdcbdebfd2c42d125a54/mink/configuration.py#L61-L62

    # Get the joint positions from MjData
    def qpos(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_addrs]

    # Set the joint velocities in MjData
    def set_qvel(self, qvel: np.ndarray, data: mujoco.MjData) -> None:
        assert qvel.size == self.qvel_qacc_addrs.size
        data.qvel[self.qvel_qacc_addrs] = qvel

    # Get the joint velocities from MjData
    def qvel(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.qvel_qacc_addrs]

    # Set the joint accelerations in MjData
    def set_qacc(self, qacc: np.ndarray, data: mujoco.MjData) -> None:
        assert qacc.size == self.qvel_qacc_addrs.size
        data.qacc[self.qvel_qacc_addrs] = qacc

    # Get the joint accelerations from MjData
    def qacc(self, data: mujoco.MjData) -> np.ndarray:
        return data.qacc[self.qvel_qacc_addrs]

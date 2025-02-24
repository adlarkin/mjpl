import mujoco
import numpy as np


# Get the joint addresses in the model's corresponding data.qpos for the specified joints.
# This assumes that `joint_ids` refer to joints with 1 DOF.
def joint_ids_to_qpos_addrs(joint_ids: list[int], model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.joint(id).qposadr.item() for id in joint_ids])


# Get the joint addresses in the model's corresponding data.qvel/data.qacc for the specified joints.
# This assumes that `joint_ids` refer to joints with 1 DOF.
def joint_ids_to_dof_addrs(joint_ids: list[int], model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.joint(id).dofadr.item() for id in joint_ids])


def joint_limits(
    joint_ids: list[int], model: mujoco.MjModel
) -> tuple[np.ndarray, np.ndarray]:
    lower_limits = np.array([model.joint(id).range[0] for id in joint_ids])
    upper_limits = np.array([model.joint(id).range[1] for id in joint_ids])
    return lower_limits, upper_limits


# Wrapper for handling a joint configuration.
# This is useful when a user is interested in interacting with a subset of the joints in MjData.
# This class assumes the joint(s) of interest have 1 DOF.
class JointGroup:
    def __init__(self, joint_ids: list[int], model: mujoco.MjModel):
        self.joint_ids = joint_ids
        self.model = model
        self.qpos_addrs = joint_ids_to_qpos_addrs(joint_ids, model)
        self.qvel_qacc_addrs = joint_ids_to_dof_addrs(joint_ids, model)
        self.lower_limits, self.upper_limits = joint_limits(joint_ids, model)

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

    def joints(self) -> list[int]:
        return self.joint_ids

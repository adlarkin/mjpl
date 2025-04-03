import mujoco
import numpy as np


def joint_ids_to_qpos_addrs(model: mujoco.MjModel, joint_ids: list[int]) -> np.ndarray:
    """Get the joint addresses for a MuJoCo model.

    This assumes that `joint_ids` refer to joints with 1 DOF.

    Args:
        model: MuJoCo model.
        joint_ids: The joints of interest from the MuJoCo model.

    Returns:
        A list of joint addresses that correspond to the joints in `joint_ids`.
        These addresses can be used with MjData.qpos
    """
    return np.array([model.joint(id).qposadr.item() for id in joint_ids])


def joint_limits(
    model: mujoco.MjModel, joint_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Get the joint limits for a MuJoCo model.

    Args:
        model: MuJoCo model.
        joint_ids: The joints of interest from the MuJoCo model.

    Returns:
        A tuple of (lower, upper) limit arrays for each joint in `joint_ids`.
    """
    lower_limits = np.array([model.joint(id).range[0] for id in joint_ids])
    upper_limits = np.array([model.joint(id).range[1] for id in joint_ids])
    return lower_limits, upper_limits


class JointGroup:
    """Class for handling a subset of joints in a MuJoCo model.

    This class assumes the joints of interest have 1 DOF.
    """

    def __init__(self, model: mujoco.MjModel, joint_ids: list[int]):
        """Constructor.

        Args:
            model: MuJoCo model.
            joint_ids: The joints of interest from the MuJoCo model.
        """
        self._joint_ids = joint_ids
        self.model = model
        self.qpos_addrs = joint_ids_to_qpos_addrs(model, joint_ids)
        self.lower_limits, self.upper_limits = joint_limits(model, joint_ids)

    def random_config(self, rng: np.random.Generator) -> np.ndarray:
        """Create a random configuration for the JointGroup.

        Args:
            rng: The random number generator.

        Returns:
            A joint configuration for the joints in the JointGroup that is
            within joint limits.
        """
        return rng.uniform(low=self.lower_limits, high=self.upper_limits)

    def fk(self, q: np.ndarray, data: mujoco.MjData) -> None:
        """Run forward kinematics on the MuJoCo data.

        Args:
            q: The joint configuration that corresponds to the joints in the JointGroup.
            data: MuJoCo data that corresponds to the JointGroup's MuJoCo model.
        """
        assert q.size == self.qpos_addrs.size
        data.qpos[self.qpos_addrs] = q
        mujoco.mj_kinematics(self.model, data)
        # NOTE: To update jacobians, a call to mj_comPos is also needed.
        # This is not being done here because this information is not needed for planning.
        # References:
        #   - https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
        #   - https://github.com/kevinzakka/mink/blob/cf1a302ff31b1f620abecdcbdebfd2c42d125a54/mink/configuration.py#L61-L62

    def qpos(self, data: mujoco.MjData) -> np.ndarray:
        """Get the configuration of the joints in the JointGroup.

        Args:
            data: MuJoCo data.

        Returns:
            The configuration of the JointGroup's joints in the MuJoCo data.
        """
        return data.qpos[self.qpos_addrs]

    @property
    def joint_ids(self) -> list[int]:
        """The joints associated with the JointGroup."""
        return self._joint_ids.copy()

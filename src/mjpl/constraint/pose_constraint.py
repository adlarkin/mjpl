import mujoco
import numpy as np
from mink import SE3
from mink.lie.so3 import RollPitchYaw

from .. import utils
from .constraint_interface import Constraint
from .joint_limit_constraint import JointLimitConstraint


class PoseConstraint(Constraint):
    """Constraint that enforces pose constraints on a site.

    This is done through a projection technique that's described in section 4a/4b here:
    https://personalrobotics.cs.washington.edu/publications/berenson2009cbirrt.pdf
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site: str,
        reference_frame: SE3,
        x_translation: tuple[float, float] = (-np.inf, np.inf),
        y_translation: tuple[float, float] = (-np.inf, np.inf),
        z_translation: tuple[float, float] = (-np.inf, np.inf),
        roll: tuple[float, float] = (-np.inf, np.inf),
        pitch: tuple[float, float] = (-np.inf, np.inf),
        yaw: tuple[float, float] = (-np.inf, np.inf),
        tolerance: float = 0.001,
        q_step: float = 0.05,
    ) -> None:
        """
        Constructor.

        Args:
            model: MuJoCo model.
            site: The site (i.e., frame) that must obey the pose constraints.
            reference_frame: The frame the pose constraints are defined relative to.
                This should be defined w.r.t. the world frame.
            x_translation: (min, max) allowed translation along the x-axis, in meters.
            y_translation: (min, max) allowed translation along the y-axis, in meters.
            z_translation: (min, max) allowed translation along the z-axis, in meters.
            roll: (min, max) allowed roll, in radians.
            pitch: (min, max) allowed pitch, in radians.
            yaw: (min, max) allowed yaw, in radians.
            tolerance: The maximum allowed deviation `site` may have from the
                pose constraints.
            q_step: The maximum distance (in configuration space) the constrained
                configuration can be from another configuration.
        """
        if tolerance < 0.0:
            raise ValueError("`tolerance` must be >= 0.")
        if q_step <= 0.0:
            raise ValueError("`q_step` must be > 0.")

        self.model = model
        self.C = np.array(
            [x_translation, y_translation, z_translation, roll, pitch, yaw]
        )
        # reference_frame is world_T_C (C = constraint frame)
        # For more information about the notation being used here, see:
        # https://manipulation.csail.mit.edu/pick.html#monogram
        self.C_T_world = reference_frame.inverse()
        self.site = site
        self.tolerance = tolerance
        self.q_step = q_step

        self.data = mujoco.MjData(model)
        self.joint_limit_constraint = JointLimitConstraint(model)
        self.site_id = model.site(site).id

    def valid_config(self, q: np.ndarray) -> bool:
        if not self.joint_limit_constraint.valid_config(q):
            return False
        dx = self._displacement_from_constraint(q)
        return np.linalg.norm(dx) <= self.tolerance

    def apply(self, q_old: np.ndarray, q: np.ndarray) -> np.ndarray | None:
        q_projected = q.copy()
        while True:
            dx = self._displacement_from_constraint(q_projected)
            if np.linalg.norm(dx) <= self.tolerance:
                return q_projected
            J = self._get_jacobian(q_projected)
            # Use pseudo-inverse in case J is singular.
            q_err = J.T @ np.linalg.pinv(J @ J.T) @ dx
            q_projected = q_projected - q_err
            violates_limits = not self.joint_limit_constraint.valid_config(q_projected)
            extends_too_far = np.linalg.norm(q_projected - q_old) > (2 * self.q_step)
            if violates_limits or extends_too_far:
                return None

    def _displacement_from_constraint(self, q: np.ndarray) -> np.ndarray:
        """Compute the displacement between a configuration and the pose constraints.

        Args:
            q: The configuration.

        Returns:
            A 6D displacement vector: {x, y, z, r, p, y}.
        """
        self.data.qpos = q
        mujoco.mj_kinematics(self.model, self.data)

        world_T_site = utils.site_pose(self.data, self.site)
        C_T_site = self.C_T_world.multiply(world_T_site)
        rpy = C_T_site.rotation().as_rpy_radians()

        d_C = np.zeros((6,))
        d_C[:3] = C_T_site.translation()
        d_C[3:] = [rpy.roll, rpy.pitch, rpy.yaw]

        c_min = self.C[:, 0]
        c_max = self.C[:, 1]

        over = d_C > c_max
        under = d_C < c_min

        delta_X = np.zeros_like(d_C)
        delta_X[over] = d_C[over] - c_max[over]
        delta_X[under] = d_C[under] - c_min[under]

        return delta_X

    def _get_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Get the RPY Jacobian of the site.

        Args:
            q: The configuration.

        Returns:
            The RPY Jacobian.
        """
        # Get the Jacobian with respect to the site.
        # mj_kinematics updates frame transforms, and mj_comPos updates jacobians:
        # - https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
        # - https://github.com/kevinzakka/mink/blob/29cb2deb3a5cb79bcc652507ebdc80685619183b/mink/configuration.py#L61-L62
        self.data.qpos = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        jac = np.zeros((6, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site_id)

        # Apply linear transformation to get the RPY Jacobian.
        world_T_site = utils.site_pose(self.data, self.site)
        rpy = world_T_site.rotation().as_rpy_radians()
        return _e_rpy(rpy) @ jac


def _e_rpy(rpy: RollPitchYaw):
    """Linear transformation that converts angular velocity Jacobian to RPY Jacobian.

    Additional information:
    - https://ieeexplore.ieee.org/document/4399305 (appendix)
    - https://personalrobotics.cs.washington.edu/publications/berenson2009cbirrt.pdf (section 4b)
    """
    c_p = np.cos(rpy.pitch)
    c_y = np.cos(rpy.yaw)
    s_p = np.sin(rpy.pitch)
    s_y = np.sin(rpy.yaw)

    E_rpy = np.eye(6)
    E_rpy[3:6, 3:5] = np.array(
        [
            [c_y / c_p, s_y / c_p],
            [-s_y, c_p],
            [c_y * (s_p / c_p), s_y * (s_p / c_p)],
        ]
    )

    return E_rpy

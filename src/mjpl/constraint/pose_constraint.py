import mink
import mujoco
import numpy as np

from .. import utils
from .constraint_interface import Constraint
from .joint_limit_constraint import JointLimitConstraint


class PoseConstraint(Constraint):
    def __init__(
        self,
        model: mujoco.MjModel,
        x_translation: tuple[float, float],
        y_translation: tuple[float, float],
        z_translation: tuple[float, float],
        roll: tuple[float, float],
        pitch: tuple[float, float],
        yaw: tuple[float, float],
        transform: mink.SE3,
        site: str,
        tolerance: float,
        q_step: float,
    ) -> None:
        """
        C: 6x2 matrix with min values in column 0 and max vals in column 1
        transform: transform for `C`, in the world frame (i.e., what limits are
            defined w.r.t. - maybe rename this to `relative_to`?)
        site: frame to apply `limits` to
        """
        if tolerance < 0.0:
            raise ValueError("`tolerance` must be >= 0.")
        if q_step <= 0.0:
            raise ValueError("`q_step` must be > 0.")

        self.model = model
        self.C = np.array(
            [x_translation, y_translation, z_translation, roll, pitch, yaw]
        )
        self.C_T_world = transform.inverse()
        self.site = site
        self.tolerance = tolerance
        self.q_step = q_step

        self.data = mujoco.MjData(model)
        self.joint_limit_constraint = JointLimitConstraint(model)
        self.site_id = model.site(site).id

    def valid_config(self, q: np.ndarray) -> bool:
        # TODO: check for joint limits and 2*q_step here as well?
        return np.linalg.norm(self._displacement_from_constraint(q)) < self.tolerance

    def apply(self, q_old: np.ndarray, q: np.ndarray) -> np.ndarray | None:
        q_projected = q
        while True:
            dx = self._displacement_from_constraint(q_projected)
            if np.linalg.norm(dx) < self.tolerance:
                return q_projected
            J = self._get_jacobian(q_projected)
            q_err = J.T @ np.linalg.inv(J @ J.T) @ dx
            q_projected = q_projected - q_err
            violates_limits = not self.joint_limit_constraint.valid_config(q_projected)
            extends_too_far = np.linalg.norm(q_projected - q_old) > (2 * self.q_step)
            if violates_limits or extends_too_far:
                return None

    def _displacement_from_constraint(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        self.data.qpos = q
        mujoco.mj_kinematics(self.model, self.data)

        world_T_site = utils.site_pose(self.data, self.site)
        C_T_site = self.C_T_world.multiply(world_T_site)
        rpy = C_T_site.rotation().as_rpy_radians()

        d_C = np.zeros((6,))
        d_C[:3] = C_T_site.translation()
        d_C[3] = rpy.roll
        d_C[4] = rpy.pitch
        d_C[5] = rpy.yaw

        delta_X = np.zeros_like(d_C)
        for i in range(delta_X.size):
            c_min = self.C[i, 0]
            c_max = self.C[i, 1]
            if d_C[i] > c_max:
                delta_X[i] = d_C[i] - c_max
            elif d_C[i] < c_min:
                delta_X[i] = d_C[i] - c_min

        return delta_X

    def _get_jacobian(self, q: np.ndarray) -> np.ndarray:
        # Get the Jacobian with respect to the site.
        # mj_kinematics updates frame transforms, and mj_comPos updates jacobians:
        # - https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
        # - https://github.com/kevinzakka/mink/blob/29cb2deb3a5cb79bcc652507ebdc80685619183b/mink/configuration.py#L61-L62
        self.data.qpos = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        jac = np.zeros((6, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site_id)

        world_T_site = utils.site_pose(self.data, self.site)
        """
        C_T_site = self.C_T_world.multiply(world_T_site)
        rpy = C_T_site.rotation().as_rpy_radians()
        """
        rpy = world_T_site.rotation().as_rpy_radians()

        c_p = np.cos(rpy.pitch)
        c_y = np.cos(rpy.yaw)
        s_p = np.sin(rpy.pitch)
        s_y = np.sin(rpy.yaw)

        E_rpy = np.zeros((6, 6))
        E_rpy[:3, :3] = np.eye(3)
        E_rpy[3:, 3:-1] = np.array(
            [
                [c_y / c_p, s_y / c_p],
                [-s_y, c_p],
                [c_y * (s_p / c_p), s_y * (s_p / c_p)],
            ]
        )
        E_rpy[5, 5] = 1

        return E_rpy @ jac

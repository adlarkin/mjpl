import unittest
from pathlib import Path

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_TWO_DOF_BALL_XML = _MODEL_DIR / "two_dof_ball.xml"


class TestPoseConstraint(unittest.TestCase):
    def test_translation_limit(self):
        model = mujoco.MjModel.from_xml_path(_TWO_DOF_BALL_XML.as_posix())
        data = mujoco.MjData(model)
        site_name = "ball_site"

        # Use the ball's initial pose as the reference frame for pose constraints.
        mujoco.mj_kinematics(model, data)
        ball_home_pose = mjpl.site_pose(data, site_name)

        # Define a pose constraint that limits ball translation along the x-axis.
        pose_constraint = mjpl.PoseConstraint(
            model,
            site_name,
            ball_home_pose,
            x_translation=(-0.1, 0.1),
            q_step=np.inf,
        )

        # Define a configuration that violates pose constraints.
        q = np.array([0.2, 0.0])
        self.assertFalse(pose_constraint.valid_config(q))

        # Apply the pose constraint and make sure the pose limits are met.
        q_constrained = pose_constraint.apply(np.array([0.0, 0.0]), q)
        self.assertIsNotNone(q_constrained)
        np.testing.assert_allclose(
            q_constrained, np.array([0.1, 0.0]), rtol=0, atol=1e-12
        )
        self.assertTrue(pose_constraint.valid_config(q_constrained))

        # Decreasing the pose constraint's q_step means trying to constrain a configuration
        # that's too far away from a previous configuration should fail.
        pose_constraint.q_step = 1e-5
        q_constrained = pose_constraint.apply(np.array([0.0, 0.0]), q)
        self.assertIsNone(q_constrained)

    def test_rotation_limit(self):
        model = load_robot_description("ur5e_mj_description")
        data = mujoco.MjData(model)
        site_name = "attachment_site"

        # Use the ee's home pose as the reference frame for pose constraints.
        q_init = model.keyframe("home").qpos
        data.qpos = q_init
        mujoco.mj_kinematics(model, data)
        ee_init_pose = mjpl.site_pose(data, site_name)
        ee_init_rpy = ee_init_pose.rotation().as_rpy_radians()

        # Randomly generate a configuration without enforcing pose constraints.
        q_rand = mjpl.random_config(
            model,
            q_init,
            mjpl.all_joints(model),
            seed=123,
            constraints=[
                mjpl.JointLimitConstraint(model),
                mjpl.CollisionConstraint(model),
            ],
        )

        # Define a pose constraint that limits the end-effector's roll and pitch.
        rotation_limit = (-0.1, 0.1)
        pose_constraint = mjpl.PoseConstraint(
            model,
            site_name,
            ee_init_pose,
            roll=rotation_limit,
            pitch=rotation_limit,
            q_step=np.inf,
        )

        # Ensure the randomly generated configuration does not obey the pose constraint.
        self.assertFalse(pose_constraint.valid_config(q_rand))

        # Apply the pose constraint to the configuration.
        q_constrained = pose_constraint.apply(q_init, q_rand)
        self.assertIsNotNone(q_constrained)
        self.assertTrue(pose_constraint.valid_config(q_constrained))

        # Compare the constrained pose to the constraint reference frame to verify
        # roll and pitch limits are met (both of these poses are expressed w.r.t.
        # the world frame, which allows us to do this direct comparison).
        data.qpos = q_constrained
        mujoco.mj_kinematics(model, data)
        constrained_pose = mjpl.site_pose(data, site_name)
        constrained_rpy = constrained_pose.rotation().as_rpy_radians()
        self.assertGreaterEqual(
            constrained_rpy.roll, ee_init_rpy.roll + rotation_limit[0]
        )
        self.assertLessEqual(constrained_rpy.roll, ee_init_rpy.roll + rotation_limit[1])
        self.assertGreaterEqual(
            constrained_rpy.pitch, ee_init_rpy.pitch + rotation_limit[0]
        )
        self.assertLessEqual(
            constrained_rpy.pitch, ee_init_rpy.pitch + rotation_limit[1]
        )

        # Decreasing the pose constraint's q_step means trying to constrain a configuration
        # that's too far away from a previous configuration should fail.
        small_q_step = 1e-5
        self.assertGreater(np.linalg.norm(q_constrained - q_init), small_q_step)
        pose_constraint.q_step = small_q_step
        q_constrained = pose_constraint.apply(q_init, q_rand)
        self.assertIsNone(q_constrained)


if __name__ == "__main__":
    unittest.main()

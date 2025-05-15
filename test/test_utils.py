import unittest
from pathlib import Path

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_BALL_XY_PLANE_XML = _MODEL_DIR / "two_dof_ball.xml"
_JOINTS_XML = _MODEL_DIR / "joints.xml"


class TestUtils(unittest.TestCase):
    def test_all_joints(self):
        model = load_robot_description("ur5e_mj_description")
        expected_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.assertListEqual(mjpl.all_joints(model), expected_joints)

        model = model.from_xml_path(_JOINTS_XML.as_posix())
        expected_joints = [
            "slide_joint",
            "free_joint",
            "hinge_joint",
            "ball_joint",
        ]
        self.assertListEqual(mjpl.all_joints(model), expected_joints)

    def test_site_pose(self):
        model = load_robot_description("ur5e_mj_description")
        data = mujoco.MjData(model)

        mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
        mujoco.mj_kinematics(model, data)

        site_name = "attachment_site"
        pose = mjpl.site_pose(data, site_name)

        site = data.site(site_name)
        np.testing.assert_allclose(site.xpos, pose.translation(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            site.xmat.reshape(3, 3), pose.rotation().as_matrix(), rtol=0, atol=1e-12
        )

    def test_qpos_idx(self):
        model = mujoco.MjModel.from_xml_path(_JOINTS_XML.as_posix())

        # Querying all joints in the model should correspond to the full mujoco.MjData.qpos
        indices = mjpl.qpos_idx(model, mjpl.all_joints(model))
        self.assertListEqual(indices, list(range(model.nq)))

        indices = mjpl.qpos_idx(model, ["slide_joint"])
        self.assertListEqual(indices, [0])

        indices = mjpl.qpos_idx(model, ["free_joint"])
        self.assertListEqual(indices, [1, 2, 3, 4, 5, 6, 7])

        indices = mjpl.qpos_idx(model, ["hinge_joint"])
        self.assertListEqual(indices, [8])

        indices = mjpl.qpos_idx(model, ["ball_joint"])
        self.assertListEqual(indices, [9, 10, 11, 12])

        # Make sure index order matches order of joints in the query.
        indices = mjpl.qpos_idx(model, ["ball_joint", "hinge_joint", "free_joint"])
        self.assertListEqual(indices, [9, 10, 11, 12, 8, 1, 2, 3, 4, 5, 6, 7])

        self.assertListEqual(mjpl.qpos_idx(model, []), [])

    def test_qvel_idx(self):
        model = mujoco.MjModel.from_xml_path(_JOINTS_XML.as_posix())

        # Querying all joints in the model should correspond to the full mujoco.MjData.qvel
        indices = mjpl.qvel_idx(model, mjpl.all_joints(model))
        self.assertListEqual(indices, list(range(model.nv)))

        indices = mjpl.qvel_idx(model, ["slide_joint"])
        self.assertListEqual(indices, [0])

        indices = mjpl.qvel_idx(model, ["free_joint"])
        self.assertListEqual(indices, [1, 2, 3, 4, 5, 6])

        indices = mjpl.qvel_idx(model, ["hinge_joint"])
        self.assertListEqual(indices, [7])

        indices = mjpl.qvel_idx(model, ["ball_joint"])
        self.assertListEqual(indices, [8, 9, 10])

        # Make sure index order matches order of joints in the query.
        indices = mjpl.qvel_idx(model, ["ball_joint", "hinge_joint", "free_joint"])
        self.assertListEqual(indices, [8, 9, 10, 7, 1, 2, 3, 4, 5, 6])

        self.assertListEqual(mjpl.qvel_idx(model, []), [])

    def test_random_config(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XY_PLANE_XML.as_posix())
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]

        seed = 42

        # Using the same seed should give the same result across multiple calls if all
        # other args are consistent.
        joints = mjpl.all_joints(model)
        q_init = np.zeros(model.nq)
        q_rand_first = mjpl.random_config(model, q_init, joints, seed, constraints)
        q_rand_second = mjpl.random_config(model, q_init, joints, seed, constraints)
        np.testing.assert_equal(q_rand_first, q_rand_second)
        self.assertTrue(mjpl.obeys_constraints(q_rand_first, constraints))
        self.assertTrue(mjpl.obeys_constraints(q_rand_second, constraints))

        # Specifying a subset of joints means some values in q_init shouldn't be modified.
        q_init = np.zeros(model.nq)
        modifiable_joints = ["ball_slide_y"]
        q_rand = mjpl.random_config(model, q_init, modifiable_joints, seed, constraints)
        unchanged_idx = mjpl.qpos_idx(model, ["ball_slide_x"])
        np.testing.assert_equal(q_rand[unchanged_idx], q_init[unchanged_idx])
        self.assertTrue(mjpl.obeys_constraints(q_rand, constraints))


if __name__ == "__main__":
    unittest.main()

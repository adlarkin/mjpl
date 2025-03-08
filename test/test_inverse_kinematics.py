import unittest

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

from mj_maniPlan.collision_ruleset import CollisionRuleset
import mj_maniPlan.inverse_kinematics as ik
from mj_maniPlan.joint_group import JointGroup


class TestInverseKinematics(unittest.TestCase):
    def test_ik(self):
        model = load_robot_description("ur5e_mj_description")
        site_name = "attachment_site"

        data = mujoco.MjData(model)
        site = data.site(site_name)
        arm_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        arm_joint_ids = [model.joint(joint).id for joint in arm_joints]
        jg = JointGroup(model, arm_joint_ids)
        cr = CollisionRuleset(model)
        q_init_world = model.keyframe("home").qpos

        # Generate a pose for the IK problem by running FK on a
        # random configuration within joint limits.
        rng = np.random.default_rng(seed=12345)
        data.qpos = rng.uniform(*model.jnt_range.T)
        mujoco.mj_kinematics(model, data)
        target_pos = site.xpos.copy()
        target_rot = site.xmat.copy()
        quat_target = np.zeros(4)
        mujoco.mju_mat2Quat(quat_target, target_rot)

        # Solve IK.
        opts = ik.IKOptions(
            jg=jg, cr=cr, pos_tolerance=1e-3, ori_tolerance=1e-3, seed=12345
        )
        q_candidate = ik.solve_ik(
            site_name, q_init_world, target_pos, target_rot.reshape(3, 3), opts
        )
        self.assertIsNotNone(q_candidate)

        # Confirm that the IK solution gives a site pose within the specified error.
        # We cannot rely on checking similarity between q_candidate and the randomly
        # generated config since there may be multiple solutions to the IK problem!
        data.qpos = q_candidate.copy()
        mujoco.mj_kinematics(model, data)
        pos = site.xpos.copy()
        pos_error = np.linalg.norm(pos - target_pos)
        self.assertLessEqual(pos_error, opts.pos_tolerance)
        rot = site.xmat.copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot)
        ori_error = np.zeros(3)
        mujoco.mju_subQuat(ori_error, quat, quat_target)
        ori_error = np.linalg.norm(ori_error)
        self.assertLessEqual(ori_error, opts.ori_tolerance)


if __name__ == "__main__":
    unittest.main()

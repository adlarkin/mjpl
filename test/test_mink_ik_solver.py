import unittest

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

from mj_maniPlan.collision_ruleset import CollisionRuleset
from mj_maniPlan.inverse_kinematics.mink_ik_solver import MinkIKSolver
from mj_maniPlan.joint_group import JointGroup
from mj_maniPlan.utils import site_pose


class TestInverseKinematics(unittest.TestCase):
    def test_ik(self):
        model = load_robot_description("ur5e_mj_description")
        site_name = "attachment_site"

        data = mujoco.MjData(model)
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
        q_init_world = model.keyframe("home").qpos.copy()

        # Generate a pose for the IK problem by running FK on a
        # random configuration within joint limits.
        rng = np.random.default_rng(seed=12345)
        data.qpos = rng.uniform(*model.jnt_range.T)
        mujoco.mj_kinematics(model, data)
        target_pose = site_pose(data, site_name)

        # Solve IK.
        pos_tolerance = 1e-3
        ori_tolerance = 1e-3
        solver = MinkIKSolver(
            model=model,
            jg=jg,
            cr=cr,
            q_init_guess=q_init_world,
            seed=12345,
        )
        q_candidate = solver.solve_ik(
            pose=target_pose,
            site=site_name,
            pos_tolerance=pos_tolerance,
            ori_tolerance=ori_tolerance,
        )
        self.assertIsNotNone(q_candidate)

        # Confirm that the IK solution gives a site pose within the specified error.
        # We cannot rely on checking similarity between q_candidate and the randomly
        # generated config since there may be multiple solutions to the IK problem!
        data.qpos = q_candidate
        mujoco.mj_kinematics(model, data)
        site = data.site(site_name)
        pos_error = np.linalg.norm(site.xpos - target_pose.translation())
        self.assertLessEqual(pos_error, pos_tolerance)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, site.xmat)
        ori_error = np.zeros(3)
        mujoco.mju_subQuat(ori_error, quat, target_pose.rotation().wxyz)
        ori_error = np.linalg.norm(ori_error)
        self.assertLessEqual(ori_error, ori_tolerance)


if __name__ == "__main__":
    unittest.main()

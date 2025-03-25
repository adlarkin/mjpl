import unittest

import mujoco
import numpy as np
from mink.lie import SE3
from robot_descriptions.loaders.mujoco import load_robot_description

import mj_maniPlan.utils as utils
from mj_maniPlan.cartesian_planner import cartesian_plan
from mj_maniPlan.collision_ruleset import CollisionRuleset
from mj_maniPlan.inverse_kinematics.mink_ik_solver import MinkIKSolver
from mj_maniPlan.joint_group import JointGroup


class TestCartesianPlanner(unittest.TestCase):
    def test_cartesian_path(self):
        model = load_robot_description("ur5e_mj_description")
        data = mujoco.MjData(model)
        site = "attachment_site"

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

        # Use the "home" keyframe as the initial configuration.
        home_keyframe = model.keyframe("home")
        q_init_world = home_keyframe.qpos.copy()

        # From the initial configuration, define a few EE poses that define the
        # desired cartesian path.
        mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
        mujoco.mj_kinematics(model, data)
        current_ee_pose = utils.site_pose(data, site)
        next_ee_pose = current_ee_pose.multiply(
            SE3.from_translation(np.array([0.02, 0.0, 0.0]))
        )
        final_ee_pose = next_ee_pose.multiply(
            SE3.from_translation(np.array([0.0, 0.02, 0.0]))
        )
        poses = [next_ee_pose, final_ee_pose]

        # Define an IK solver.
        pos_tolerance = 1e-3
        ori_tolerance = 1e-3
        solver = MinkIKSolver(
            model=model,
            jg=jg,
            cr=cr,
            pos_tolerance=pos_tolerance,
            ori_tolerance=ori_tolerance,
            seed=12345,
            max_attempts=5,
        )

        # Plan a cartesian path.
        path = cartesian_plan(q_init_world, poses, site, solver)
        self.assertEqual(len(path), 3)
        # The first element in the path should match the initial configuration.
        np.testing.assert_equal(path[0], q_init_world)
        # The other joint configurations in the path should satisfy the poses
        # within the IK solver's tolerance.
        data.qpos = path[1]
        mujoco.mj_kinematics(model, data)
        actual_site_pose = utils.site_pose(data, site)
        err = poses[0].minus(actual_site_pose)
        self.assertLessEqual(np.linalg.norm(err[:3]), pos_tolerance)
        self.assertLessEqual(np.linalg.norm(err[3:]), ori_tolerance)
        data.qpos = path[2]
        mujoco.mj_kinematics(model, data)
        actual_site_pose = utils.site_pose(data, site)
        err = poses[1].minus(actual_site_pose)
        self.assertLessEqual(np.linalg.norm(err[:3]), pos_tolerance)
        self.assertLessEqual(np.linalg.norm(err[3:]), ori_tolerance)


if __name__ == "__main__":
    unittest.main()

import unittest

import mujoco
import numpy as np
from dm_control import mjcf
from robot_descriptions import g1_mj_description
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl


class TestMinkIKSolver(unittest.TestCase):
    def test_ik(self):
        model = load_robot_description("ur5e_mj_description")
        site_name = "attachment_site"
        cr = mjpl.CollisionRuleset()

        q_init_world = model.keyframe("home").qpos.copy()

        # Generate a pose for the IK problem by running FK on a
        # random configuration within joint limits.
        data = mujoco.MjData(model)
        rng = np.random.default_rng(seed=12345)
        data.qpos = rng.uniform(*model.jnt_range.T)
        mujoco.mj_kinematics(model, data)
        target_pose = mjpl.site_pose(data, site_name)

        pos_tolerance = 1e-3
        ori_tolerance = 1e-3
        solver = mjpl.MinkIKSolver(
            model=model,
            joints=mjpl.all_joints(model),
            cr=cr,
            pos_tolerance=pos_tolerance,
            ori_tolerance=ori_tolerance,
            seed=12345,
            max_attempts=5,
        )

        # Solve IK (test both with/without an initial guess config)
        ik_solutions = [
            solver.solve_ik(
                pose=target_pose,
                site=site_name,
                q_init_guess=q_init_world,
            ),
            solver.solve_ik(
                pose=target_pose,
                site=site_name,
                q_init_guess=None,
            ),
        ]
        self.assertIsNotNone(ik_solutions[0])
        self.assertIsNotNone(ik_solutions[1])

        # Confirm that the IK solution gives a site pose within the specified error.
        # We cannot rely on checking similarity between q_candidate and the randomly
        # generated config since there may be multiple solutions to the IK problem!
        for solution in ik_solutions:
            data.qpos = solution.copy()
            mujoco.mj_kinematics(model, data)
            actual_site_pose = mjpl.site_pose(data, site_name)
            err = target_pose.minus(actual_site_pose)
            self.assertLessEqual(np.linalg.norm(err[:3]), pos_tolerance)
            self.assertLessEqual(np.linalg.norm(err[3:]), ori_tolerance)

    def test_ik_subset_joints(self):
        # Use a humanoid model, which has a high DOF.
        # This allows testing IK on only a subset of the joints.
        mjcf_model = mjcf.from_path(g1_mj_description.MJCF_PATH)

        # Add a site to the humanoid's left wrist which can be used for IK.
        wrist_body = mjcf_model.find("body", "left_wrist_yaw_link")
        wrist_body.add("site", name="ee_site")

        model = mujoco.MjModel.from_xml_string(
            mjcf_model.to_xml_string(), mjcf_model.get_assets()
        )
        left_arm_joints = [
            "left_elbow_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
            "left_wrist_yaw_joint",
        ]

        import pdb

        pdb.set_trace()
        q_rand = mjpl.random_valid_config(
            model,
            mujoco.MjData(model).qpos,
            left_arm_joints,
            seed=42,
        )

        # TODO: finish this:
        # - extract target pose form q_rand
        # - solve IK with both left_arm_joints and all joints
        #   - make sure that for left_arm_joints, no other joints are modified
        # - another option is to generate q_rand for all joints, and test that
        #   IK fails if only left_arm_joints are used (assuming the pose for q_rand
        #   requires other joints to be modified in order for the pose to be reached)

    def test_invalid_args(self):
        model = load_robot_description("ur5e_mj_description")
        cr = mjpl.CollisionRuleset()
        joints = mjpl.all_joints(model)

        with self.assertRaisesRegex(ValueError, "`max_attempts` must be > 0"):
            mjpl.MinkIKSolver(
                model=model,
                joints=joints,
                cr=cr,
                max_attempts=-2,
            )
            mjpl.MinkIKSolver(
                model=model,
                joints=joints,
                cr=cr,
                max_attempts=0,
            )

        with self.assertRaisesRegex(ValueError, "`iterations` must be > 0"):
            mjpl.MinkIKSolver(
                model=model,
                joints=joints,
                cr=cr,
                iterations=-2,
            )
            mjpl.MinkIKSolver(
                model=model,
                joints=joints,
                cr=cr,
                iterations=0,
            )

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            mjpl.MinkIKSolver(
                model=model,
                joints=[],
            )


if __name__ == "__main__":
    unittest.main()

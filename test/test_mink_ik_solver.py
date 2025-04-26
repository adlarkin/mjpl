import unittest

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl


class TestMinkIKSolver(unittest.TestCase):
    def setUp(self):
        self.model = load_robot_description("ur5e_mj_description")
        self.site_name = "attachment_site"
        self.cr = mjpl.CollisionRuleset()

    def test_ik(self):
        q_init_world = self.model.keyframe("home").qpos.copy()

        # Generate a pose for the IK problem by running FK on a
        # random configuration within joint limits.
        data = mujoco.MjData(self.model)
        rng = np.random.default_rng(seed=12345)
        data.qpos = rng.uniform(*self.model.jnt_range.T)
        mujoco.mj_kinematics(self.model, data)
        target_pose = mjpl.site_pose(data, self.site_name)

        pos_tolerance = 1e-3
        ori_tolerance = 1e-3
        solver = mjpl.MinkIKSolver(
            model=self.model,
            joints=mjpl.all_joints(self.model),
            cr=self.cr,
            pos_tolerance=pos_tolerance,
            ori_tolerance=ori_tolerance,
            seed=12345,
            max_attempts=5,
        )

        # Solve IK (test both with/without an initial guess config)
        ik_solutions = [
            solver.solve_ik(
                pose=target_pose,
                site=self.site_name,
                q_init_guess=q_init_world,
            ),
            solver.solve_ik(
                pose=target_pose,
                site=self.site_name,
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
            mujoco.mj_kinematics(self.model, data)
            actual_site_pose = mjpl.site_pose(data, self.site_name)
            err = target_pose.minus(actual_site_pose)
            self.assertLessEqual(np.linalg.norm(err[:3]), pos_tolerance)
            self.assertLessEqual(np.linalg.norm(err[3:]), ori_tolerance)

    def test_invalid_args(self):
        joints = mjpl.all_joints(self.model)

        with self.assertRaisesRegex(ValueError, "`max_attempts` must be > 0"):
            mjpl.MinkIKSolver(
                model=self.model,
                joints=joints,
                cr=self.cr,
                max_attempts=-2,
            )
            mjpl.MinkIKSolver(
                model=self.model,
                joints=joints,
                cr=self.cr,
                max_attempts=0,
            )

        with self.assertRaisesRegex(ValueError, "`iterations` must be > 0"):
            mjpl.MinkIKSolver(
                model=self.model,
                joints=joints,
                cr=self.cr,
                iterations=-2,
            )
            mjpl.MinkIKSolver(
                model=self.model,
                joints=joints,
                cr=self.cr,
                iterations=0,
            )

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            mjpl.MinkIKSolver(
                model=self.model,
                joints=[],
            )


if __name__ == "__main__":
    unittest.main()

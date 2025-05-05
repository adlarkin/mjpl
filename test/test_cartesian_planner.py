import unittest

import mujoco
import numpy as np
from mink.lie import SE3, SO3
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl
from mjpl.planning.cartesian_planner import _interpolate_poses


def poses_approximately_equal(
    p1: SE3, p2: SE3, pos_tolerance: float = 1e-9, ori_tolerance: float = 1e-9
) -> None:
    np.testing.assert_allclose(
        p1.translation(), p2.translation(), rtol=0.0, atol=pos_tolerance
    )
    np.testing.assert_allclose(
        p1.rotation().parameters(),
        p2.rotation().parameters(),
        rtol=0.0,
        atol=ori_tolerance,
    )


class TestCartesianPlanner(unittest.TestCase):
    def test_interpolate_pose(self):
        start_pose = SE3.from_rotation_and_translation(
            SO3.from_x_radians(0), np.array([0, 0, 0])
        )
        end_pose = SE3.from_rotation_and_translation(
            SO3.from_x_radians(np.pi), np.array([1, 0, 0])
        )

        # Thresholds of inf mean no interpolation occurs, so we should just get
        # the start and end pose
        poses = _interpolate_poses(
            start_pose, end_pose, lin_threshold=np.inf, ori_threshold=np.inf
        )
        self.assertEqual(len(poses), 2)
        self.assertEqual(poses[0], start_pose)
        self.assertEqual(poses[1], end_pose)

        # Interpolate based on position
        poses = _interpolate_poses(
            start_pose, end_pose, lin_threshold=0.65, ori_threshold=np.inf
        )
        self.assertEqual(len(poses), 3)
        self.assertEqual(poses[0], start_pose)
        self.assertEqual(poses[2], end_pose)
        halfway_pose = SE3.from_rotation_and_translation(
            SO3.from_x_radians(np.pi / 2), np.array([0.5, 0.0, 0.0])
        )
        poses_approximately_equal(poses[1], halfway_pose)

        # Interpolate based on orientation
        poses = _interpolate_poses(
            start_pose, end_pose, lin_threshold=np.inf, ori_threshold=np.pi * 0.3
        )
        self.assertEqual(len(poses), 5)
        self.assertEqual(poses[0], start_pose)
        self.assertEqual(poses[4], end_pose)
        # There should be three evenly spaced intermediate poses
        intermediate_poses = [
            SE3.from_rotation_and_translation(
                SO3.from_x_radians(np.pi * 0.25), np.array([0.25, 0.0, 0.0])
            ),
            SE3.from_rotation_and_translation(
                SO3.from_x_radians(np.pi * 0.5), np.array([0.5, 0.0, 0.0])
            ),
            SE3.from_rotation_and_translation(
                SO3.from_x_radians(np.pi * 0.75), np.array([0.75, 0.0, 0.0])
            ),
        ]
        poses_approximately_equal(poses[1], intermediate_poses[0])
        poses_approximately_equal(poses[2], intermediate_poses[1])
        poses_approximately_equal(poses[3], intermediate_poses[2])

        # If thresholds for position and orientation are given, the one that
        # requires more interpolation steps should take preference.
        # In this scenario, we should get the same behavior as the scenario
        # above (where only orientation was applied) since orientation requires
        # more interpolation steps than linear.
        poses = _interpolate_poses(
            start_pose, end_pose, lin_threshold=0.65, ori_threshold=np.pi * 0.3
        )
        self.assertEqual(len(poses), 5)
        self.assertEqual(poses[0], start_pose)
        self.assertEqual(poses[4], end_pose)
        poses_approximately_equal(poses[1], intermediate_poses[0])
        poses_approximately_equal(poses[2], intermediate_poses[1])
        poses_approximately_equal(poses[3], intermediate_poses[2])

        with self.assertRaisesRegex(ValueError, "`lin_threshold` must be > 0"):
            _interpolate_poses(
                start_pose, end_pose, lin_threshold=0.0, ori_threshold=np.inf
            )
            _interpolate_poses(
                start_pose, end_pose, lin_threshold=-1.0, ori_threshold=np.inf
            )
        with self.assertRaisesRegex(ValueError, "`ori_threshold` must be > 0"):
            _interpolate_poses(
                start_pose, end_pose, lin_threshold=np.inf, ori_threshold=0.0
            )
            _interpolate_poses(
                start_pose, end_pose, lin_threshold=np.inf, ori_threshold=-1.0
            )

    def test_cartesian_path(self):
        model = load_robot_description("ur5e_mj_description")
        data = mujoco.MjData(model)
        site = "attachment_site"

        # Use the "home" keyframe as the initial configuration.
        home_keyframe = model.keyframe("home")
        q_init = home_keyframe.qpos.copy()

        # From the initial configuration, define a few EE poses that define the
        # desired Cartesian path.
        mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
        mujoco.mj_kinematics(model, data)
        current_ee_pose = mjpl.site_pose(data, site)
        next_ee_pose = current_ee_pose.multiply(
            SE3.from_translation(np.array([0.02, 0.0, 0.0]))
        )
        final_ee_pose = next_ee_pose.multiply(
            SE3.from_translation(np.array([0.0, 0.02, 0.0]))
        )
        poses = [next_ee_pose, final_ee_pose]

        interpolated_pose = current_ee_pose.multiply(
            SE3.from_translation(np.array([0.02, 0.01, 0.0]))
        )

        pos_tolerance = 1e-3
        ori_tolerance = 1e-3
        solver = mjpl.MinkIKSolver(
            model=model,
            joints=mjpl.all_joints(model),
            constraints=[mjpl.CollisionConstraint(model)],
            pos_tolerance=pos_tolerance,
            ori_tolerance=ori_tolerance,
            seed=12345,
            max_attempts=5,
        )

        # Plan a Cartesian path.
        waypoints = mjpl.cartesian_plan(
            q_init,
            poses,
            site,
            solver,
            lin_threshold=0.01,
            ori_threshold=0.1,
        )
        self.assertEqual(len(waypoints), 4)

        # The first element in the path should match the initial configuration.
        np.testing.assert_equal(waypoints[0], q_init)

        # The other joint configurations in the path should satisfy the poses
        # within the IK solver's tolerance.
        data.qpos = waypoints[1]
        mujoco.mj_kinematics(model, data)
        actual_site_pose = mjpl.site_pose(data, site)
        err = poses[0].minus(actual_site_pose)
        self.assertLessEqual(np.linalg.norm(err[:3]), pos_tolerance)
        self.assertLessEqual(np.linalg.norm(err[3:]), ori_tolerance)

        # An interpolated pose should have been added to the Cartesian path
        data.qpos = waypoints[2]
        mujoco.mj_kinematics(model, data)
        actual_site_pose = mjpl.site_pose(data, site)
        err = interpolated_pose.minus(actual_site_pose)
        self.assertLessEqual(np.linalg.norm(err[:3]), pos_tolerance)
        self.assertLessEqual(np.linalg.norm(err[3:]), ori_tolerance)
        data.qpos = waypoints[3]
        mujoco.mj_kinematics(model, data)
        actual_site_pose = mjpl.site_pose(data, site)
        err = poses[1].minus(actual_site_pose)
        self.assertLessEqual(np.linalg.norm(err[:3]), pos_tolerance)
        self.assertLessEqual(np.linalg.norm(err[3:]), ori_tolerance)


if __name__ == "__main__":
    unittest.main()

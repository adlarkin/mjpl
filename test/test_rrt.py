import unittest
from pathlib import Path

import mujoco
import numpy as np

import mjpl

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_ONE_DOF_BALL_XML = _MODEL_DIR / "one_dof_ball.xml"
_TWO_DOF_BALL_XML = _MODEL_DIR / "two_dof_ball.xml"


class TestRRT(unittest.TestCase):
    def test_run_rrt(self):
        model = mujoco.MjModel.from_xml_path(_ONE_DOF_BALL_XML.as_posix())
        cr = mjpl.CollisionRuleset()
        epsilon = 0.1

        q_init = np.array([-0.2])
        q_goal = np.array([0.35])

        planner = mjpl.RRT(
            model,
            mjpl.all_joints(model),
            cr,
            max_planning_time=5.0,
            epsilon=epsilon,
            seed=42,
        )
        path = planner.plan_to_config(q_init, q_goal)
        self.assertIsNotNone(path)
        np.testing.assert_equal(path.q_init, q_init)
        self.assertGreater(len(path.waypoints), 2)
        self.assertListEqual(path.joints, mjpl.all_joints(model))

        # The path should start at q_init and end at q_goal.
        np.testing.assert_equal(path.waypoints[0], q_init)
        np.testing.assert_equal(path.waypoints[-1], q_goal)

        # Subsequent waypoints should be no further than epsilon apart.
        for i in range(1, len(path.waypoints)):
            self.assertLessEqual(
                np.linalg.norm(path.waypoints[i] - path.waypoints[i - 1]),
                epsilon,
            )

        # The path's initial state and all waypoints should be valid configs.
        data = mujoco.MjData(model)
        data.qpos = path.q_init
        self.assertTrue(mjpl.utils.is_valid_config(model, data, cr))
        for wp in path.waypoints:
            data.qpos = wp
            self.assertTrue(mjpl.utils.is_valid_config(model, data, cr))

    def test_run_rrt_subset_joints(self):
        model = mujoco.MjModel.from_xml_path(_TWO_DOF_BALL_XML.as_posix())
        cr = mjpl.CollisionRuleset()
        epsilon = 0.1

        q_init = np.array([0.0, 0.0])
        q_goal = np.array([0.3, 0.0])
        planning_joints = ["ball_slide_x"]
        q_idx = mjpl.qpos_idx(model, planning_joints)

        planner = mjpl.RRT(
            model,
            planning_joints,
            cr,
            max_planning_time=5.0,
            epsilon=epsilon,
            seed=42,
        )
        path = planner.plan_to_config(q_init, q_goal)
        self.assertIsNotNone(path)
        np.testing.assert_equal(path.q_init, q_init)
        self.assertGreater(len(path.waypoints), 2)
        self.assertListEqual(path.joints, planning_joints)

        # The path should start at q_init and end at q_goal.
        np.testing.assert_equal(path.waypoints[0], q_init[q_idx])
        np.testing.assert_equal(path.waypoints[-1], q_goal[q_idx])

        # Subsequent waypoints should be no further than epsilon apart.
        for i in range(1, len(path.waypoints)):
            prev = path.q_init.copy()
            prev[q_idx] = path.waypoints[i - 1]
            next = path.q_init.copy()
            next[q_idx] = path.waypoints[i]
            self.assertLessEqual(np.linalg.norm(next - prev), epsilon)

        # The path's initial state and all waypoints should be valid configs.
        data = mujoco.MjData(model)
        data.qpos = path.q_init
        self.assertTrue(mjpl.utils.is_valid_config(model, data, cr))
        for wp in path.waypoints:
            data.qpos[q_idx] = wp
            self.assertTrue(mjpl.utils.is_valid_config(model, data, cr))

    def test_trivial_rrt(self):
        model = mujoco.MjModel.from_xml_path(_ONE_DOF_BALL_XML.as_posix())
        cr = mjpl.CollisionRuleset()
        epsilon = 0.1

        q_init = np.array([0.0])
        q_goal = np.array([0.05])

        # Plan to a goal that is immediately reachable.
        planner = mjpl.RRT(
            model,
            mjpl.all_joints(model),
            cr,
            max_planning_time=5.0,
            epsilon=epsilon,
            seed=42,
        )
        path = planner.plan_to_config(q_init, q_goal)
        self.assertIsNotNone(path)
        self.assertEqual(len(path.waypoints), 2)
        np.testing.assert_equal(path.waypoints[0], q_init)
        np.testing.assert_equal(path.waypoints[1], q_goal)
        self.assertListEqual(path.joints, mjpl.all_joints(model))

    def test_trivial_rrt_subset_joints(self):
        model = mujoco.MjModel.from_xml_path(_TWO_DOF_BALL_XML.as_posix())
        cr = mjpl.CollisionRuleset()
        epsilon = 0.1

        q_init = np.array([0.0, 0.0])
        q_goal = np.array([0.05, 0.0])
        planning_joints = ["ball_slide_x"]
        q_idx = mjpl.qpos_idx(model, planning_joints)

        # Plan to a goal that is immediately reachable.
        planner = mjpl.RRT(
            model,
            planning_joints,
            cr,
            max_planning_time=5.0,
            epsilon=epsilon,
            seed=42,
        )
        path = planner.plan_to_config(q_init, q_goal)
        self.assertIsNotNone(path)
        self.assertEqual(len(path.waypoints), 2)
        np.testing.assert_equal(path.waypoints[0], q_init[q_idx])
        np.testing.assert_equal(path.waypoints[1], q_goal[q_idx])
        self.assertListEqual(path.joints, planning_joints)

    def test_invalid_args(self):
        model = mujoco.MjModel.from_xml_path(_ONE_DOF_BALL_XML.as_posix())
        joints = mjpl.all_joints(model)
        cr = mjpl.CollisionRuleset()

        with self.assertRaisesRegex(ValueError, "max_planning_time"):
            mjpl.RRT(model, joints, cr, max_planning_time=0.0)
            mjpl.RRT(model, joints, cr, max_planning_time=-1.0)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            mjpl.RRT(model, joints, cr, epsilon=0.0)
            mjpl.RRT(model, joints, cr, epsilon=-1.0)
        with self.assertRaisesRegex(ValueError, "max_connection_distance"):
            mjpl.RRT(model, joints, cr, max_connection_distance=0.0)
            mjpl.RRT(model, joints, cr, max_connection_distance=-1.0)
        with self.assertRaisesRegex(ValueError, "goal_biasing_probability"):
            mjpl.RRT(model, joints, cr, goal_biasing_probability=-1.0)
            mjpl.RRT(model, joints, cr, goal_biasing_probability=2.0)
        with self.assertRaisesRegex(ValueError, "planning_joints"):
            mjpl.RRT(model, [], cr)

        model = mujoco.MjModel.from_xml_path(_TWO_DOF_BALL_XML.as_posix())
        joints = ["ball_slide_y"]
        planner = mjpl.RRT(
            model,
            ["ball_slide_y"],
            mjpl.CollisionRuleset(),
            max_planning_time=5.0,
            epsilon=0.1,
            seed=42,
        )
        with self.assertRaisesRegex(
            ValueError, "values for joints outside of the planner's planning joints"
        ):
            planner.plan_to_config(np.array([0.0, 0.0]), np.array([0.1, 0.0]))


if __name__ == "__main__":
    unittest.main()

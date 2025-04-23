import unittest
from pathlib import Path

import mujoco
import numpy as np

import mjpl

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_BALL_XML = _MODEL_DIR / "one_dof_ball.xml"


class TestRRT(unittest.TestCase):
    def test_run_rrt(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XML.as_posix())
        cr = mjpl.CollisionRuleset(model)
        epsilon = 0.1

        q_init = np.array([-0.2])
        q_goal = np.array([0.35])

        # Planning for all joints can be done by explicitly listing all joints, or by
        # using an empty list for `planning_joints`.
        for planning_joints in (["ball_slide_x"], []):
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
            q_idx = mjpl.qpos_idx(model, planning_joints, default_to_full=True)
            for wp in path.waypoints:
                data.qpos[q_idx] = wp
                self.assertTrue(mjpl.utils.is_valid_config(model, data, cr))

    def test_trivial_rrt(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XML.as_posix())
        cr = mjpl.CollisionRuleset(model)
        epsilon = 0.1

        q_init = np.array([0.0])
        q_goal = np.array([0.05])

        # Plan to a goal that is immediately reachable.
        planner = mjpl.RRT(
            model, [], cr, max_planning_time=5.0, epsilon=epsilon, seed=42
        )
        path = planner.plan_to_config(q_init, q_goal)
        self.assertIsNotNone(path)
        self.assertEqual(len(path.waypoints), 2)
        np.testing.assert_equal(path.waypoints[0], q_init)
        np.testing.assert_equal(path.waypoints[1], q_goal)
        self.assertListEqual(path.joints, [])

    def test_invalid_args(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XML.as_posix())
        cr = mjpl.CollisionRuleset(model)

        with self.assertRaisesRegex(ValueError, "max_planning_time"):
            mjpl.RRT(model, [], cr, max_planning_time=0.0)
            mjpl.RRT(model, [], cr, max_planning_time=-1.0)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            mjpl.RRT(model, [], cr, epsilon=0.0)
            mjpl.RRT(model, [], cr, epsilon=-1.0)
        with self.assertRaisesRegex(ValueError, "max_connection_distance"):
            mjpl.RRT(model, [], cr, max_connection_distance=0.0)
            mjpl.RRT(model, [], cr, max_connection_distance=-1.0)
        with self.assertRaisesRegex(ValueError, "goal_biasing_probability"):
            mjpl.RRT(model, [], cr, goal_biasing_probability=-1.0)
            mjpl.RRT(model, [], cr, goal_biasing_probability=2.0)


if __name__ == "__main__":
    unittest.main()

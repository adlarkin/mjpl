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

        planner = mjpl.RRT(
            model, ["ball_slide_x"], cr, max_planning_time=5.0, epsilon=epsilon, seed=42
        )
        path = planner.plan_to_config(q_init, q_goal)
        self.assertIsNotNone(path)
        np.testing.assert_equal(path.q_init, q_init)
        self.assertGreater(len(path.waypoints), 2)
        self.assertListEqual(path.joints, ["ball_slide_x"])

        # The path should start at q_init and end at q_goal.
        np.testing.assert_equal(path.waypoints[0], q_init)
        np.testing.assert_equal(path.waypoints[-1], q_goal)

        for i in range(1, len(path.waypoints)):
            self.assertLessEqual(
                np.linalg.norm(path.waypoints[i] - path.waypoints[i - 1]),
                epsilon,
            )

    def test_trivial_rrt(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XML.as_posix())
        cr = mjpl.CollisionRuleset(model)
        epsilon = 0.1

        q_init = np.array([0.0])
        q_goal = np.array([0.05])

        # Plan to a goal that is immediately reachable.
        planner = mjpl.RRT(
            model, ["ball_slide_x"], cr, max_planning_time=5.0, epsilon=epsilon, seed=42
        )
        path = planner.plan_to_config(q_init, q_goal)
        self.assertIsNotNone(path)
        self.assertEqual(len(path.waypoints), 2)
        np.testing.assert_equal(path.waypoints[0], q_init)
        np.testing.assert_equal(path.waypoints[1], q_goal)

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

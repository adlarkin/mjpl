import unittest
from pathlib import Path

import mujoco
import numpy as np

import mjpl

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_BALL_XML = _MODEL_DIR / "ball.xml"


class TestRRT(unittest.TestCase):
    def test_run_rrt(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XML.as_posix())
        jg = mjpl.JointGroup(model, [model.joint("ball_slide_x").id])
        cr = mjpl.CollisionRuleset(model)
        epsilon = 0.1

        q_init = np.array([-0.2])
        q_goal = np.array([0.35])

        planner = mjpl.RRT(jg, cr, max_planning_time=5.0, epsilon=epsilon, seed=42)
        path = planner.plan_to_config(q_init, q_goal)
        self.assertGreater(len(path), 2)

        # The path should start at q_init and end at q_goal.
        np.testing.assert_equal(path[0], q_init)
        np.testing.assert_equal(path[-1], q_goal)

        for i in range(1, len(path)):
            self.assertLessEqual(
                np.linalg.norm(path[i] - path[i - 1]),
                epsilon,
            )

    def test_trivial_rrt(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XML.as_posix())
        jg = mjpl.JointGroup(model, [model.joint("ball_slide_x").id])
        cr = mjpl.CollisionRuleset(model)
        epsilon = 0.1

        q_init = np.array([0.0])
        q_goal = np.array([0.05])

        # Plan to a goal that is immediately reachable.
        planner = mjpl.RRT(jg, cr, max_planning_time=5.0, epsilon=epsilon, seed=42)
        path = planner.plan_to_config(q_init, q_goal)
        self.assertEqual(len(path), 2)
        np.testing.assert_equal(path[0], q_init)
        np.testing.assert_equal(path[1], q_goal)


if __name__ == "__main__":
    unittest.main()

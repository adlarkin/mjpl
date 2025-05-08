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
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]
        epsilon = 0.1

        q_init = np.array([-0.2])
        q_goal = np.array([0.35])

        planner = mjpl.RRT(
            model,
            mjpl.all_joints(model),
            constraints,
            max_planning_time=5.0,
            epsilon=epsilon,
            seed=42,
        )
        waypoints = planner.plan_to_config(q_init, q_goal)
        self.assertGreater(len(waypoints), 2)

        # The waypoints should start at q_init and end at q_goal.
        np.testing.assert_equal(waypoints[0], q_init)
        np.testing.assert_equal(waypoints[-1], q_goal)

        # Subsequent waypoints should be no further than epsilon apart.
        for i in range(1, len(waypoints)):
            self.assertLessEqual(
                np.linalg.norm(waypoints[i] - waypoints[i - 1]),
                epsilon,
            )

        # The waypoints should obey constraints.
        for wp in waypoints:
            self.assertTrue(mjpl.obeys_constraints(wp, constraints))

    def test_run_rrt_subset_joints(self):
        model = mujoco.MjModel.from_xml_path(_TWO_DOF_BALL_XML.as_posix())
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]
        epsilon = 0.1

        q_init = np.array([0.0, 0.0])
        q_goal = np.array([0.3, 0.0])
        planning_joints = ["ball_slide_x"]

        planner = mjpl.RRT(
            model,
            planning_joints,
            constraints,
            max_planning_time=5.0,
            epsilon=epsilon,
            seed=42,
        )
        waypoints = planner.plan_to_config(q_init, q_goal)
        self.assertGreater(len(waypoints), 2)

        # The waypoints should start at q_init and end at q_goal.
        np.testing.assert_equal(waypoints[0], q_init)
        np.testing.assert_equal(waypoints[-1], q_goal)

        # Subsequent waypoints should be no further than epsilon apart.
        for i in range(1, len(waypoints)):
            self.assertLessEqual(
                np.linalg.norm(waypoints[i] - waypoints[i - 1]),
                epsilon,
            )

        # The waypoints should obey constraints.
        for wp in waypoints:
            self.assertTrue(mjpl.obeys_constraints(wp, constraints))

    def test_trivial_rrt(self):
        model = mujoco.MjModel.from_xml_path(_ONE_DOF_BALL_XML.as_posix())
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]
        epsilon = 0.1

        q_init = np.array([0.0])
        q_goal = np.array([0.05])

        # Plan to a goal that is immediately reachable.
        planner = mjpl.RRT(
            model,
            mjpl.all_joints(model),
            constraints,
            max_planning_time=5.0,
            epsilon=epsilon,
            seed=42,
        )
        waypoints = planner.plan_to_config(q_init, q_goal)
        self.assertEqual(len(waypoints), 2)
        np.testing.assert_equal(waypoints[0], q_init)
        np.testing.assert_equal(waypoints[1], q_goal)

        # The waypoints should obey constraints.
        self.assertTrue(mjpl.obeys_constraints(waypoints[0], constraints))
        self.assertTrue(mjpl.obeys_constraints(waypoints[1], constraints))

    def test_trivial_rrt_subset_joints(self):
        model = mujoco.MjModel.from_xml_path(_TWO_DOF_BALL_XML.as_posix())
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]
        epsilon = 0.1

        q_init = np.array([0.0, 0.0])
        q_goal = np.array([0.05, 0.0])
        planning_joints = ["ball_slide_x"]

        # Plan to a goal that is immediately reachable.
        planner = mjpl.RRT(
            model,
            planning_joints,
            constraints,
            max_planning_time=5.0,
            epsilon=epsilon,
            seed=42,
        )
        waypoints = planner.plan_to_config(q_init, q_goal)
        self.assertEqual(len(waypoints), 2)
        np.testing.assert_equal(waypoints[0], q_init)
        np.testing.assert_equal(waypoints[1], q_goal)

        # The waypoints should obey constraints.
        self.assertTrue(mjpl.obeys_constraints(waypoints[0], constraints))
        self.assertTrue(mjpl.obeys_constraints(waypoints[1], constraints))

    def test_invalid_args(self):
        model = mujoco.MjModel.from_xml_path(_ONE_DOF_BALL_XML.as_posix())
        joints = mjpl.all_joints(model)

        with self.assertRaisesRegex(ValueError, "max_planning_time"):
            mjpl.RRT(model, joints, [], max_planning_time=0.0)
            mjpl.RRT(model, joints, [], max_planning_time=-1.0)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            mjpl.RRT(model, joints, [], epsilon=0.0)
            mjpl.RRT(model, joints, [], epsilon=-1.0)
        with self.assertRaisesRegex(ValueError, "goal_biasing_probability"):
            mjpl.RRT(model, joints, [], goal_biasing_probability=-1.0)
            mjpl.RRT(model, joints, [], goal_biasing_probability=2.0)
        with self.assertRaisesRegex(ValueError, "planning_joints"):
            mjpl.RRT(model, [], [])

        model = mujoco.MjModel.from_xml_path(_TWO_DOF_BALL_XML.as_posix())
        joints = ["ball_slide_y"]
        planner = mjpl.RRT(
            model,
            ["ball_slide_y"],
            [],
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

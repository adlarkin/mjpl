import unittest
from pathlib import Path

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl
from mjpl.planning.tree import Node, Tree
from mjpl.planning.utils import _combine_paths, _constrained_extend, _step

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_ONE_DOF_BALL_XML = _MODEL_DIR / "one_dof_ball.xml"
_TWO_DOF_BALL_XML = _MODEL_DIR / "two_dof_ball.xml"


def directly_connectable_waypoints() -> list[np.ndarray]:
    """
    Waypoints that can be directly connected between the start and end.
    "X" represents an obstacle.

    NOTE: This is meant to be used with the two_dof_ball.xml test file.

                  p2 -> p3 --> p4
                  ^            |
                  |     X      |
            p0 -> p1    X      |
                        X      |
                               v
                        p6 <-- p5
    """
    p0 = np.array([0.0, 0.0])
    p1 = np.array([0.25, 0.0])
    p2 = np.array([0.25, 0.75])
    p3 = np.array([0.5, 0.75])
    p4 = np.array([1.0, 0.75])
    p5 = np.array([1.0, -0.75])
    p6 = np.array([0.5, -0.75])
    return [p0, p1, p2, p3, p4, p5, p6]


def shortcuttable_waypoints() -> list[np.ndarray]:
    """
    Waypoints that can benefit from shortcutting.
    "X" represents an obstacle.

    NOTE: This is meant to be used with the two_dof_ball.xml test file.

                  p2 -> p3 --> p4
                  ^            |
                  |     X      v
            p0 -> p1    X      p5 -> p6
                        X
    """
    p0 = np.array([0.0, 0.0])
    p1 = np.array([0.25, 0.0])
    p2 = np.array([0.25, 1.5])
    p3 = np.array([0.5, 1.5])
    p4 = np.array([1.0, 1.5])
    p5 = np.array([1.0, 0.0])
    p6 = np.array([1.0, 0.0])
    return [p0, p1, p2, p3, p4, p5, p6]


class TestPlanningUtils(unittest.TestCase):
    def test_smooth_path_on_directly_connectable_path(self):
        model = mujoco.MjModel.from_xml_path(_TWO_DOF_BALL_XML.as_posix())
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]
        seed = 5
        eps = 0.1

        waypoints = directly_connectable_waypoints()

        # The first and last waypoints can be connected without violating constraints.
        shortened_waypoints = mjpl.smooth_path(
            waypoints, constraints, eps=eps, seed=seed
        )
        self.assertLess(
            mjpl.path_length(shortened_waypoints), mjpl.path_length(waypoints)
        )
        self.assertGreater(len(shortened_waypoints), 2)
        np.testing.assert_equal(shortened_waypoints[0], waypoints[0])
        np.testing.assert_equal(shortened_waypoints[-1], waypoints[-1])
        for i in range(1, len(shortened_waypoints)):
            # Add tolerance to the epsilon check to account for floating point error.
            self.assertLessEqual(
                np.linalg.norm(shortened_waypoints[i] - shortened_waypoints[i - 1]),
                eps + 1e-8,
            )
        for wp in shortened_waypoints:
            self.assertTrue(mjpl.obeys_constraints(wp, constraints))

        # Run smooting on the same path with sparse=True.
        shortened_waypoints = mjpl.smooth_path(
            waypoints, constraints, eps=eps, seed=seed, sparse=True
        )
        self.assertLess(
            mjpl.path_length(shortened_waypoints), mjpl.path_length(waypoints)
        )
        self.assertListEqual(shortened_waypoints, [waypoints[0], waypoints[-1]])

    def test_smooth_path_around_obstacle(self):
        model = mujoco.MjModel.from_xml_path(_TWO_DOF_BALL_XML.as_posix())
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]
        seed = 5
        eps = 0.1

        waypoints = shortcuttable_waypoints()

        smoothed_waypoints = mjpl.smooth_path(
            waypoints, constraints, eps=eps, seed=seed
        )
        self.assertLess(
            mjpl.path_length(smoothed_waypoints), mjpl.path_length(waypoints)
        )
        self.assertGreater(len(smoothed_waypoints), 2)
        np.testing.assert_equal(smoothed_waypoints[0], waypoints[0])
        np.testing.assert_equal(smoothed_waypoints[-1], waypoints[-1])
        for i in range(1, len(smoothed_waypoints)):
            # Add tolerance to the epsilon check to account for floating point error.
            self.assertLessEqual(
                np.linalg.norm(smoothed_waypoints[i] - smoothed_waypoints[i - 1]),
                eps + 1e-8,
            )
        for wp in smoothed_waypoints:
            self.assertTrue(mjpl.obeys_constraints(wp, constraints))

        # Run smooting on the same path with sparse=True. Since there's an obstacle in
        # the way, the smoothed path should have more than just the start/end waypoint.
        smoothed_waypoints = mjpl.smooth_path(
            waypoints, constraints, eps=eps, seed=seed, sparse=True
        )
        self.assertLess(
            mjpl.path_length(smoothed_waypoints), mjpl.path_length(waypoints)
        )
        self.assertGreater(len(smoothed_waypoints), 2)
        np.testing.assert_equal(smoothed_waypoints[0], waypoints[0])
        np.testing.assert_equal(smoothed_waypoints[-1], waypoints[-1])
        for wp in smoothed_waypoints:
            self.assertTrue(mjpl.obeys_constraints(wp, constraints))

    def test_smooth_path_6dof(self):
        model = load_robot_description("ur5e_mj_description")
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]
        seed = 42

        # Make a path starting from the home config that connects to various random valid configs.
        waypoints = [model.keyframe("home").qpos.copy()]
        unique_waypoints = {tuple(waypoints[0])}
        for i in range(5):
            q_rand = mjpl.random_config(
                model, np.zeros(model.nq), mjpl.all_joints(model), seed + i, constraints
            )
            waypoints.append(q_rand)
            hashable_q_rand = tuple(q_rand)
            self.assertNotIn(hashable_q_rand, unique_waypoints)
            unique_waypoints.add(tuple(q_rand))

        # Smooth the path.
        smoothed_waypoints = mjpl.smooth_path(waypoints, constraints, seed=seed)
        self.assertLess(
            mjpl.path_length(smoothed_waypoints), mjpl.path_length(waypoints)
        )
        np.testing.assert_equal(smoothed_waypoints[0], waypoints[0])
        np.testing.assert_equal(smoothed_waypoints[-1], waypoints[-1])
        for wp in smoothed_waypoints:
            self.assertTrue(mjpl.obeys_constraints(wp, constraints))

    def test_path_length(self):
        waypoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ]
        self.assertAlmostEqual(mjpl.path_length(waypoints), 3.0)

    def test_constrained_extend_that_reaches_target(self):
        model = mujoco.MjModel.from_xml_path(_ONE_DOF_BALL_XML.as_posix())
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]
        epsilon = 0.1

        q_init = np.array([-0.1])
        tree = Tree(Node(q_init))

        # Test a constrained extend that should reach the target.
        q_goal = np.array([0.15])
        q_reached = _constrained_extend(q_goal, tree, epsilon, constraints)
        np.testing.assert_equal(q_reached, q_goal)

        # Check the path from the last connected node.
        # This implicitly checks each node's parent.
        expected_path = [
            q_goal,
            np.array([0.1]),
            np.array([0.0]),
            q_init,
        ]
        path = [n.q for n in tree.get_path(tree.nearest_neighbor(q_goal))]
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            np.testing.assert_allclose(path[i], expected_path[i], rtol=0, atol=1e-9)

    def test_constrained_extend_that_violates_constraint(self):
        model = mujoco.MjModel.from_xml_path(_ONE_DOF_BALL_XML.as_posix())
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]
        epsilon = 0.1

        q_init = np.array([0.0])
        tree = Tree(Node(q_init))

        obstacle = model.geom("wall_obstacle")
        obstacle_min_x = obstacle.pos[0] - obstacle.size[0]

        # Test a constrained extend that cannot reach the target due to constraint
        # violation (in this case, collision). q_reached should be just before the obstacle.
        q_goal = np.array([1.0])
        q_reached = _constrained_extend(q_goal, tree, epsilon, constraints)
        np.testing.assert_array_less(q_init, q_reached)
        np.testing.assert_array_less(q_reached, obstacle_min_x)
        np.testing.assert_array_less(q_reached, q_goal)

    def test_constrained_extend_towards_existing_config(self):
        model = mujoco.MjModel.from_xml_path(_ONE_DOF_BALL_XML.as_posix())
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]
        epsilon = 0.1

        root_node = Node(np.array([0.0]))
        tree = Tree(root_node)

        # Running constrained extend on a configuration that's already in the tree
        # should do nothing.
        q_reached = _constrained_extend(root_node.q, tree, epsilon, constraints)
        np.testing.assert_equal(q_reached, root_node.q)
        self.assertSetEqual(tree.nodes, {root_node})

    def test_step(self):
        start = np.array([0.0, 0.0])
        target = np.array([0.5, 0.0])

        q_next = _step(start, target, max_step_dist=5.0)
        np.testing.assert_equal(q_next, target)

        q_next = _step(start, target, max_step_dist=0.1)
        np.testing.assert_allclose(q_next, np.array([0.1, 0.0]), rtol=0, atol=1e-8)

        q_next = _step(target, target, max_step_dist=np.inf)
        np.testing.assert_equal(q_next, target)

        with self.assertRaisesRegex(ValueError, "`max_step_dist` must be > 0.0"):
            _step(start, target, max_step_dist=0.0)
            _step(start, target, max_step_dist=-1.0)

    def test_combine_paths(self):
        root_start = Node(np.array([0.0]))
        child_start = Node(np.array([0.1]), parent=root_start)
        start_tree = Tree(root_start)
        start_tree.add_node(child_start)

        root_goal = Node(np.array([0.3]))
        child_goal = Node(np.array([0.2]), parent=root_goal)
        goal_tree = Tree(root_goal)
        goal_tree.add_node(child_goal)

        expected_path = [
            root_start.q,
            child_start.q,
            child_goal.q,
            root_goal.q,
        ]
        path = _combine_paths(start_tree, child_start, goal_tree, child_goal)
        self.assertListEqual(path, expected_path)

        # Add a duplicate "merge node".
        q_new = np.array([0.15])
        grandchild_start = Node(q_new, parent=child_start)
        start_tree.add_node(grandchild_start)
        grandchild_goal = Node(q_new, parent=child_goal)
        goal_tree.add_node(grandchild_goal)

        # Make sure the duplicate node is properly handled when combining paths.
        expected_path = [
            root_start.q,
            child_start.q,
            q_new,
            child_goal.q,
            root_goal.q,
        ]
        path = _combine_paths(start_tree, grandchild_start, goal_tree, grandchild_goal)
        self.assertListEqual(path, expected_path)


if __name__ == "__main__":
    unittest.main()

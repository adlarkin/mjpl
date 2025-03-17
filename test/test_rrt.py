import unittest
from pathlib import Path

import mujoco
import numpy as np

import mj_maniPlan.rrt as rrt
import mj_maniPlan.utils as utils
from mj_maniPlan.collision_ruleset import CollisionRuleset
from mj_maniPlan.joint_group import JointGroup

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_BALL_XML = _MODEL_DIR / "ball.xml"


class TestRRT(unittest.TestCase):
    def load_ball_with_obstacle_model(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XML.as_posix())

        self.obstacle = model.geom("wall_obstacle")

        planning_joints = [model.joint("ball_slide_x").id]
        jg = JointGroup(model, planning_joints)

        cr = CollisionRuleset(model)

        options = rrt.RRTOptions(
            jg=jg,
            cr=cr,
            max_planning_time=5.0,
            epsilon=0.1,
            seed=42,
        )

        # Initial joint configuration.
        # We're testing with a simple model: a ball that can slide along the x-axis.
        # So there's only one value in data.qpos (the ball's x position)
        self.q_init = np.array([-0.1])

        self.planner = rrt.RRT(options)

    def test_extend(self):
        self.load_ball_with_obstacle_model()

        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        q_goal = np.array([0.5])
        expected_q_extended = np.array([0.0])
        extended_node = self.planner._extend(q_goal, tree)
        self.assertIsNotNone(extended_node)
        self.assertEqual(len(tree.nodes), 2)
        self.assertAlmostEqual(extended_node.q, expected_q_extended, places=9)

        # Check the path starting from the extended node.
        # This implicitly checks the extended node's parent.
        expected_path = [
            expected_q_extended,
            self.q_init,
        ]
        path = tree.get_path(extended_node)
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            self.assertAlmostEqual(path[i][0], expected_path[i][0], places=9)

        # extending towards a q that is already in the tree should do nothing
        existing_nodes = tree.nodes.copy()
        same_extended_node = self.planner._extend(extended_node.q, tree)
        self.assertIsNotNone(same_extended_node)
        self.assertSetEqual(tree.nodes, existing_nodes)
        # The assertSetEqual above runs the equality operator on all elements of the set.
        # Since node equality is defined as having the same q, we should take the check a
        # step further to ensure that same_extended_node and extended_node are the same object.
        self.assertIs(same_extended_node, extended_node)

        # Test extend where the "nearest node" is already given
        tree = rrt.Tree()
        root_node = rrt.Node(np.array([0.0]), None)
        tree.add_node(root_node)
        tree.add_node(rrt.Node(np.array([2.0]), root_node))
        extended_node = self.planner._extend(np.array([3.0]), tree, root_node)
        self.assertIsNotNone(extended_node)
        self.assertIs(extended_node.parent, root_node)
        self.assertAlmostEqual(extended_node.q[0], 0.1, places=9)

    def test_connect(self):
        self.load_ball_with_obstacle_model()

        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        q_goal = np.array([0.15])
        goal_node = self.planner._connect(q_goal, tree)

        expected_qs_in_tree = [
            self.q_init,
            np.array([0.0]),
            np.array([0.1]),
            q_goal,
        ]
        for q_tree in expected_qs_in_tree:
            n = tree.nearest_neighbor(q_tree)
            self.assertAlmostEqual(n.q[0], q_tree[0], places=9)

        # Check the path from the last connected node.
        # This implicitly checks each connected node's parent.
        expected_path = expected_qs_in_tree[::-1]
        path = tree.get_path(goal_node)
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            self.assertAlmostEqual(path[i][0], expected_path[i][0], places=9)

        # connecting towards a q that is already in the tree should do nothing
        existing_nodes = tree.nodes.copy()
        same_connected_node = self.planner._connect(goal_node.q, tree)
        self.assertSetEqual(tree.nodes, existing_nodes)
        # The assertSetEqual above runs the equality operator on all elements of the set.
        # Since node equality is defined as having the same q, we should take the check a
        # step further to ensure that same_extended_node and extended_node are the same object.
        self.assertIs(same_connected_node, goal_node)

    def test_connect_max_distance(self):
        self.load_ball_with_obstacle_model()

        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))
        q_goal = np.array([0.15])

        # Run one connect operation to a max distance.
        goal_node = self.planner._connect(
            q_goal, tree, eps=0.05, max_connection_distance=0.075
        )

        expected_qs_in_tree = [
            self.q_init,
            np.array([-0.05]),  # Moves one "eps"
            np.array([-0.025]),  # Moves to max connection distance
        ]
        for q_tree in expected_qs_in_tree:
            n = tree.nearest_neighbor(q_tree)
            self.assertAlmostEqual(n.q[0], q_tree[0], places=9)

        # Run a second and third connect which should still not get to the goal.
        for _ in range(2):
            goal_node = self.planner._connect(
                q_goal, tree, eps=0.05, max_connection_distance=0.075
            )

        expected_qs_in_tree.extend(
            [
                np.array([0.025]),  # Moves one "eps"
                np.array([0.05]),  # Moves to max connection distance
                np.array([0.1]),  # Moves one more "eps"
                np.array([0.125]),  # Moves to max connection distance
            ]
        )
        for q_tree in expected_qs_in_tree:
            n = tree.nearest_neighbor(q_tree)
            self.assertAlmostEqual(n.q[0], q_tree[0], places=9)

        # The final connect should finally reach the goal!
        goal_node = self.planner._connect(
            q_goal, tree, eps=0.05, max_connection_distance=0.075
        )
        self.assertAlmostEqual(goal_node.q[0], q_goal[0], places=9)

    def test_extend_and_connect_into_obstacle(self):
        self.load_ball_with_obstacle_model()

        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        # Try to connect to the maximum joint value for the ball.
        # There is an obstacle in the way, so the final connection should be just before
        # where the obstacle lies.
        q_goal = np.array([1.0])
        connected_node = self.planner._connect(q_goal, tree)
        obstacle_min_x = self.obstacle.pos[0] - self.obstacle.size[0]
        self.assertLess(connected_node.q[0], obstacle_min_x)

        # Try to extend towards the goal.
        # This should not work since the extension is in collision.
        existing_nodes = tree.nodes.copy()
        extended_node = self.planner._extend(q_goal, tree)
        self.assertIsNone(extended_node)
        self.assertSetEqual(tree.nodes, existing_nodes)

    def test_get_path(self):
        self.load_ball_with_obstacle_model()

        q_new = np.array([0.15])

        start_tree = rrt.Tree()
        start_tree.add_node(rrt.Node(self.q_init, None))
        connected_node_a = self.planner._connect(q_new, start_tree)

        goal_tree = rrt.Tree()
        goal_tree.add_node(rrt.Node(np.array([0.5]), None))
        connected_node_b = self.planner._connect(q_new, goal_tree)

        path = self.planner.get_path(
            start_tree, connected_node_a, goal_tree, connected_node_b
        )
        expected_path = [
            self.q_init,
            np.array([0.0]),
            np.array([0.1]),
            np.array([0.15]),
            np.array([0.2]),
            np.array([0.3]),
            np.array([0.4]),
            np.array([0.5]),
        ]
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            self.assertAlmostEqual(path[i][0], expected_path[i][0], places=9)

    def test_run_rrt(self):
        self.load_ball_with_obstacle_model()

        q_goal = np.array([0.35])
        path = self.planner.plan_to_config(self.q_init, q_goal)
        self.assertGreater(len(path), 2)

        # The path should start at q_init and end at q_goal
        np.testing.assert_equal(path[0], self.q_init)
        np.testing.assert_equal(path[-1], q_goal)

        for i in range(1, len(path)):
            self.assertLessEqual(
                utils.configuration_distance(path[i - 1], path[i]),
                self.planner.options.epsilon,
            )

    def test_trivial_rrt(self):
        self.load_ball_with_obstacle_model()

        # If we plan to a goal that is directly reachable, the planner should make the direct connection and exit
        q_goal = np.array([-0.05])
        path = self.planner.plan_to_config(self.q_init, q_goal)
        self.assertEqual(len(path), 2)
        np.testing.assert_equal(path[0], self.q_init)
        np.testing.assert_equal(path[1], q_goal)


if __name__ == "__main__":
    unittest.main()

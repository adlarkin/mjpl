import mujoco
import numpy as np
import os
import unittest

from mj_maniPlan.sampling import HaltonSampler
import mj_maniPlan.rrt as rrt


class TestRRT(unittest.TestCase):
    def setUp(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        model_file = dir + "/models/ball.xml"
        model = mujoco.MjModel.from_xml_path(model_file)

        joint_names = [ "ball_slide_x" ]
        self.options = rrt.RRTOptions(
            joint_names=joint_names,
            max_planning_time=5.0,
            epsilon=0.1,
            rng=HaltonSampler(len(joint_names), seed=42)
        )

        # Set the initial joint configuration.
        # We're testing with a simple model: a ball that can slide along the x-axis.
        # So there's only one value in data.qpos (the ball's x position)
        self.q_init = np.array([-0.1])
        data = mujoco.MjData(model)
        data.qpos[0] = self.q_init[0]
        mujoco.mj_kinematics(model, data)

        self.rrt = rrt.RRT(self.options, model, data)

    def test_extend(self):
        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        q_goal = np.array([0.5])
        self.rrt.extend(q_goal, tree)
        self.assertEqual(len(tree.nodes), 2)

        expected_q_extended = np.array([0.0])
        extended_node = tree.nearest_neighbor(expected_q_extended)
        self.assertAlmostEqual(extended_node.q, expected_q_extended, places=9)

        # Check the path starting from the extended node.
        # This implicitly checks the extended node's parent.
        expected_path = [
            expected_q_extended,
            self.q_init,
        ]
        tree.set_path_root(extended_node)
        path = tree.get_path()
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            self.assertAlmostEqual(path[i][0], expected_path[i][0], places=9)

        # extending towards a q that is already in the tree should do nothing
        existing_nodes = tree.nodes.copy()
        self.rrt.extend(extended_node.q, tree)
        self.assertSetEqual(tree.nodes, existing_nodes)

        # TODO: test the nearest_node arg of the extend API

    def test_connect(self):
        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        q_goal = np.array([0.15])
        self.rrt.connect(q_goal, tree)

        expected_qs_in_tree = [
            self.q_init,
            np.array([0.0]),
            np.array([0.1]),
            q_goal,
        ]
        for q_tree in expected_qs_in_tree:
            n = tree.nearest_neighbor(q_tree)
            self.assertAlmostEqual(n.q[0], q_tree[0], places=9)

        goal_node = tree.nearest_neighbor(q_goal)

        # Check the path from the last connected node.
        # This implicitly checks each connected node's parent.
        expected_path = expected_qs_in_tree[::-1]
        tree.set_path_root(goal_node)
        path = tree.get_path()
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            self.assertAlmostEqual(path[i][0], expected_path[i][0], places=9)

        # connecting towards a q that is already in the tree should do nothing
        existing_nodes = tree.nodes.copy()
        self.rrt.connect(goal_node.q, tree)
        self.assertSetEqual(tree.nodes, existing_nodes)

    def test_get_path(self):
        q_new = np.array([0.15])

        start_tree = rrt.Tree()
        start_tree.add_node(rrt.Node(self.q_init, None))
        self.rrt.connect(q_new, start_tree)
        start_tree.set_path_root(start_tree.nearest_neighbor(q_new))

        goal_tree = rrt.Tree()
        goal_tree.add_node(rrt.Node(np.array([0.5]), None))
        self.rrt.connect(q_new, goal_tree)
        goal_tree.set_path_root(goal_tree.nearest_neighbor(q_new))

        path = self.rrt.get_path(start_tree, goal_tree)
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
        q_goal = np.array([0.35])
        path = self.rrt.plan(q_goal)
        self.assertIsNotNone(path)

        # The path should start at q_init and end at q_goal
        self.assertTrue(np.array_equal(path[0], self.q_init))
        self.assertTrue(np.array_equal(path[-1], q_goal))

        # The path should be strictly increasing to q_goal
        for i in range(1, len(path)):
            self.assertGreater(path[i][0], path[i-1][0])

    # TODO: test with an obstacle?
    # (for extend and connect - can add something like a plane to the simple model)


if __name__ == '__main__':
    unittest.main()

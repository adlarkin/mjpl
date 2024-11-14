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

        self.obstacle = model.geom('wall_obstacle')

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
        expected_q_extended = np.array([0.0])
        extended_node = self.rrt.extend(q_goal, tree)
        self.assertIsNotNone(extended_node)
        self.assertEqual(len(tree.nodes), 2)
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
        same_extended_node = self.rrt.extend(extended_node.q, tree)
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
        extended_node = self.rrt.extend(np.array([3.0]), tree, root_node)
        self.assertIsNotNone(extended_node)
        self.assertIs(extended_node.parent, root_node)
        self.assertAlmostEqual(extended_node.q[0], 0.1, places=9)

    def test_connect(self):
        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        q_goal = np.array([0.15])
        goal_node = self.rrt.connect(q_goal, tree)

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
        tree.set_path_root(goal_node)
        path = tree.get_path()
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            self.assertAlmostEqual(path[i][0], expected_path[i][0], places=9)

        # connecting towards a q that is already in the tree should do nothing
        existing_nodes = tree.nodes.copy()
        same_connected_node = self.rrt.connect(goal_node.q, tree)
        self.assertSetEqual(tree.nodes, existing_nodes)
        # The assertSetEqual above runs the equality operator on all elements of the set.
        # Since node equality is defined as having the same q, we should take the check a
        # step further to ensure that same_extended_node and extended_node are the same object.
        self.assertIs(same_connected_node, goal_node)

    def test_extend_and_connect_into_obstacle(self):
        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        # Try to connect to the maximum joint value for the ball.
        # There is an obstacle in the way, so the final connection should be just before
        # where the obstacle lies.
        q_goal = np.array([1.0])
        connected_node = self.rrt.connect(q_goal, tree)
        obstacle_min_x = self.obstacle.pos[0] - self.obstacle.size[0]
        self.assertLess(connected_node.q[0], obstacle_min_x)

        # Try to extend towards the goal.
        # This should not work since the extension is in collision.
        existing_nodes = tree.nodes.copy()
        extended_node = self.rrt.extend(q_goal, tree)
        self.assertIsNone(extended_node)
        self.assertSetEqual(tree.nodes, existing_nodes)

    def test_get_path(self):
        q_new = np.array([0.15])

        start_tree = rrt.Tree()
        start_tree.add_node(rrt.Node(self.q_init, None))
        connected_node_a = self.rrt.connect(q_new, start_tree)
        start_tree.set_path_root(connected_node_a)

        goal_tree = rrt.Tree()
        goal_tree.add_node(rrt.Node(np.array([0.5]), None))
        connected_node_b = self.rrt.connect(q_new, goal_tree)
        goal_tree.set_path_root(connected_node_b)

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

    def test_trivial_rrt(self):
        # If we plan to a goal that is directly reachable, the planner should make the direct connection and exit
        q_goal = np.array([-0.05])
        path = self.rrt.plan(q_goal)
        self.assertEqual(len(path), 2)
        self.assertTrue(np.array_equal(path[0], self.q_init))
        self.assertTrue(np.array_equal(path[1], q_goal))

    def test_shortcut(self):
        # TODO: clean up this test case, setUp does a lot of similar things
        # but for the 1 DOF ball model with an obstacle
        dir = os.path.dirname(os.path.realpath(__file__))
        model_file = dir + "/models/ball_xy_plane.xml"
        model = mujoco.MjModel.from_xml_path(model_file)

        joint_names = [
            "ball_slide_x",
            "ball_slide_y",
        ]
        options = rrt.RRTOptions(
            joint_names=joint_names,
            max_planning_time=5.0,
            epsilon=0.1,
            rng=HaltonSampler(len(joint_names), seed=42)
        )

        # Set the initial joint configuration.
        q_init = np.array([-0.1, 0.0])
        data = mujoco.MjData(model)
        data.qpos = q_init.copy()
        mujoco.mj_kinematics(model, data)

        my_rrt = rrt.RRT(options, model, data)

        '''
        Suboptimal path that can benefit from shortcutting

                          n_3 -> n_4 -> n_5
                           ^             |
                           |             v
            n_0 -> n_1 -> n_2           n_6 -> n_7
        '''
        tree = rrt.Tree()
        n_0 = rrt.Node(q_init, None)
        n_1 = rrt.Node(np.array([0.0, 0.0]), n_0)
        n_2 = rrt.Node(np.array([0.5, 0.0]), n_1)
        n_3 = rrt.Node(np.array([0.5, 0.5]), n_2)
        n_4 = rrt.Node(np.array([1.0, 0.5]), n_3)
        n_5 = rrt.Node(np.array([1.5, 0.5]), n_4)
        n_6 = rrt.Node(np.array([1.5, 0.0]), n_5)
        n_7 = rrt.Node(np.array([2.0, 0.0]), n_6)
        nodes = [ n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7 ]
        for n in nodes:
            tree.add_node(n)

        tree.set_path_root(n_7)
        path = tree.get_path()
        path.reverse()
        self.assertEqual(len(path), len(nodes))
        for i in range(len(path)):
            self.assertTrue(np.array_equal(path[i], nodes[i].q))

        expected_shorcut_path = [
            n_0.q,
            n_1.q,
            n_2.q,
            np.array([1.0, 0.0]),
            n_6.q,
            n_7.q,
        ]
        shortcut_path = my_rrt.shortcut(path, 2, 6)
        print(shortcut_path)
        self.assertEqual(len(shortcut_path), len(expected_shorcut_path))


if __name__ == '__main__':
    unittest.main()

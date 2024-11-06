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

    # TODO: test extending towards a node that is already in the tree
    def test_extend(self):
        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        q_goal = np.array([0.5])
        self.rrt.extend(q_goal, tree)

        # since nodes are defined as equal only by their joint configurations and not their parents,
        # we don't need to specify the parent of the newly extended node (the one with q = [0.0])
        q_extended = np.array([0.0])
        expected_nodes_in_tree = {
            rrt.Node(self.q_init, None),
            rrt.Node(q_extended, None),
        }
        self.assertSetEqual(expected_nodes_in_tree, tree.nodes)

        # Check the path starting from the extended node.
        # This implicitly checks the extended node's parent.
        expected_path = [
            np.array([0.0]),
            np.array([-0.1]),
        ]
        tree.set_path_root(tree.nearest_neighbor(q_extended))
        self.assertTrue(np.array_equal(tree.get_path(), expected_path))

    # TODO: test connecting towards a node that is already in the tree
    def test_connect(self):
        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        q_goal = np.array([0.2])
        self.rrt.connect(q_goal, tree)

        # since nodes are defined as equal only by their joint configurations and not their parents,
        # we don't need to specify the parent of the connected nodes
        expected_nodes_in_tree = {
            rrt.Node(self.q_init, None),
            rrt.Node(np.array([0.0]), None),
            rrt.Node(np.array([0.1]), None),
            rrt.Node(np.array([0.2]), None),
        }
        self.assertSetEqual(expected_nodes_in_tree, tree.nodes)

        # Check the path from the last connected node.
        # This implicitly checks each connected node's parent.
        expected_path = [
            np.array([0.2]),
            np.array([0.1]),
            np.array([0.0]),
            np.array([-0.1]),
        ]
        tree.set_path_root(tree.nearest_neighbor(np.array([0.2])))
        self.assertTrue(np.array_equal(tree.get_path(), expected_path))

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
        # TODO: implement this almostEqual check in other methods too
        # (currently just doing np.arrayEqual checks, but different machines have different floating point accuracy)
        for i in range(len(path)):
            # The values in path and expected_path may not be exactly the same due to floating point error.
            self.assertAlmostEqual(path[i][0], expected_path[i][0], places=9)

    def test_run_rrt(self):
        pass


if __name__ == '__main__':
    unittest.main()

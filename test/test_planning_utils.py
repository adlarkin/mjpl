import unittest
from pathlib import Path

import mujoco
import numpy as np

import mjpl
from mjpl.planning.tree import Node, Tree
from mjpl.planning.utils import _combine_paths, _connect, _extend

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_BALL_XML = _MODEL_DIR / "ball_with_obstacle.xml"


class TestPlanningUtils(unittest.TestCase):
    def test_extend(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XML.as_posix())
        data = mujoco.MjData(model)
        jg = mjpl.JointGroup(model, [model.joint("ball_slide_x").id])
        cr = mjpl.CollisionRuleset(model)
        epsilon = 0.1

        q_init = np.array([-0.1])
        root_node = Node(q_init)
        tree = Tree(root_node)

        # Test a valid EXTEND.
        q_goal = np.array([0.5])
        expected_q_extended = np.array([0.0])
        extended_node = _extend(q_goal, tree, root_node, epsilon, jg, cr, data)
        self.assertIsNotNone(extended_node)
        self.assertSetEqual(tree.nodes, {root_node, extended_node})
        np.testing.assert_allclose(
            extended_node.q, expected_q_extended, rtol=0, atol=1e-9
        )
        self.assertTrue(extended_node.parent, root_node)

        # Running EXTEND when q_target == start_node.q should do nothing.
        existing_nodes = tree.nodes.copy()
        same_extended_node = _extend(
            extended_node.q, tree, extended_node, epsilon, jg, cr, data
        )
        self.assertIsNotNone(same_extended_node)
        self.assertIn(same_extended_node, existing_nodes)
        self.assertSetEqual(tree.nodes, existing_nodes)

        # EXTEND should fail if the result of the extension is an invalid node
        # (in this case, a node that is in collision).
        q_init = np.array([0.75])
        root_node = Node(q_init)
        tree = Tree(root_node)
        extended_node = _extend(np.array([0.9]), tree, root_node, epsilon, jg, cr, data)
        self.assertIsNone(extended_node)
        self.assertSetEqual(tree.nodes, {root_node})

    def test_connect(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XML.as_posix())
        data = mujoco.MjData(model)
        jg = mjpl.JointGroup(model, [model.joint("ball_slide_x").id])
        cr = mjpl.CollisionRuleset(model)
        epsilon = 0.1

        q_init = np.array([-0.1])
        root_node = Node(q_init)
        tree = Tree(root_node)

        # Test a valid CONNECT.
        q_goal = np.array([0.15])
        connected_node = _connect(q_goal, tree, epsilon, np.inf, jg, cr, data)
        np.testing.assert_allclose(connected_node.q, q_goal, rtol=0, atol=1e-9)
        # Check the path from the last connected node.
        # This implicitly checks each connected node's parent.
        expected_path = [
            q_goal,
            np.array([0.1]),
            np.array([0.0]),
            q_init,
        ]
        path = [n.q for n in tree.get_path(connected_node)]
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            np.testing.assert_allclose(path[i], expected_path[i], rtol=0, atol=1e-9)

        # Running CONNECT when q_target corresponds to a node that's already
        # in the tree should do nothing.
        existing_nodes = tree.nodes.copy()
        same_connected_node = _connect(
            connected_node.q, tree, epsilon, np.inf, jg, cr, data
        )
        self.assertIn(same_connected_node, existing_nodes)
        self.assertSetEqual(tree.nodes, existing_nodes)

        # Test CONNECT with a max connection distance that is < distance from q_init to q_goal.
        q_init = np.array([0.0])
        root_node = Node(q_init)
        tree = Tree(root_node)
        q_goal = np.array([0.5])
        max_connection_dist = 0.45
        max_connected_q = q_init + max_connection_dist
        connected_node = _connect(
            q_goal, tree, epsilon, max_connection_dist, jg, cr, data
        )
        np.testing.assert_allclose(connected_node.q, max_connected_q, rtol=0, atol=1e-9)
        # Check the path from the last connected node.
        # This implicitly checks each connected node's parent.
        expected_path = [
            max_connected_q,
            np.array([0.4]),
            np.array([0.3]),
            np.array([0.2]),
            np.array([0.1]),
            q_init,
        ]
        path = [n.q for n in tree.get_path(connected_node)]
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            np.testing.assert_allclose(path[i], expected_path[i], rtol=0, atol=1e-9)

        # Test CONNECT with an invalid goal (in this case, one that is in collision).
        # CONNECT should stop just before the obstacle.
        q_init = np.array([0.0])
        root_node = Node(q_init)
        tree = Tree(root_node)
        obstacle = model.geom("wall_obstacle")
        obstacle_min_x = obstacle.pos[0] - obstacle.size[0]
        q_goal = np.array([1.0])
        connected_node = _connect(q_goal, tree, epsilon, np.inf, jg, cr, data)
        self.assertNotEqual(connected_node, root_node)
        self.assertGreater(len(tree.nodes), 1)
        np.testing.assert_array_less(connected_node.q, obstacle_min_x)

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
        self.assertTrue(len(path), len(expected_path))
        for i in range(len(path)):
            np.testing.assert_array_equal(path[i], expected_path[i])

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
        self.assertTrue(len(path), len(expected_path))
        for i in range(len(path)):
            np.testing.assert_array_equal(path[i], expected_path[i])


if __name__ == "__main__":
    unittest.main()

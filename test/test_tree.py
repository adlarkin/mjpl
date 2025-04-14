import unittest

import numpy as np

from mjpl.planning.tree import Node, Tree


class TestNode(unittest.TestCase):
    def test_eq(self):
        a = Node(np.array([0, 1, 2]), None)
        b = Node(np.array([0, 1, 2]), a)
        c = Node(np.array([3, 4, 5]), a)
        d = Node(np.array([3, 4, 5]), b)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertNotEqual(b, c)
        self.assertEqual(c, d)


class TestTree(unittest.TestCase):
    def build_tree(self) -> Tree:
        tree = Tree(self.root)
        self.assertIsNone(tree.sink_node)
        nodes = [self.n_1, self.n_2, self.n_3]
        for n in nodes:
            tree.add_node(n)
        self.assertEqual(len(tree.nodes), 4)
        return tree

    def setUp(self):
        """
        The nodes defined here will form the following tree:

                        n_2
                         ^
                         |
                        root -> n_1 -> n_3
        """
        self.root = Node(np.array([0, 0]), None)
        self.n_1 = Node(np.array([1, 0]), self.root)
        self.n_2 = Node(np.array([0, 1]), self.root)
        self.n_3 = Node(np.array([2, 0]), self.n_1)

    def test_create_tree(self):
        root = Node(np.array([0]))

        with self.assertRaisesRegex(ValueError, "root node should have no parent"):
            Tree(Node(q=np.array([1]), parent=root))

        tree = Tree(root)
        self.assertIn(root, tree)

    def test_add_node(self):
        tree = self.build_tree()

        # Don't add a node that's already in the tree
        existing_node = Node(self.n_2.q, self.n_3)
        self.assertIn(existing_node, tree)
        with self.assertRaisesRegex(ValueError, "already exists in the tree"):
            tree.add_node(existing_node)

        # Don't add a node that has no parent
        no_parent_node = Node(np.array([5, 5]))
        self.assertNotIn(no_parent_node, tree)
        with self.assertRaisesRegex(ValueError, "Node does not have a parent"):
            tree.add_node(no_parent_node)

        # Don't add a node if the parent is not in the tree
        other_root_node = Node(np.array([10, 0]))
        self.assertNotIn(other_root_node, tree)
        wrong_tree_node = Node(np.array([20, 5]), other_root_node)
        self.assertNotIn(wrong_tree_node, tree)
        with self.assertRaisesRegex(ValueError, "parent is not in the tree"):
            tree.add_node(wrong_tree_node)

        # Add a node that's not already in the tree
        new_node = Node(np.array([-1, -1]), parent=self.n_1)
        self.assertNotIn(new_node, tree)
        tree.add_node(new_node)
        self.assertIn(new_node, tree)

    def test_nearest_neighbor(self):
        tree = self.build_tree()

        nn = tree.nearest_neighbor(np.array([2, 1]))
        self.assertEqual(nn, self.n_3)

        # check nearest neighbor for a q that is equidistant to multiple nodes in the tree
        self.assertIn(tree.nearest_neighbor(np.array([1, 1])), {self.n_1, self.n_2})

        # check nearest neighbor for a q that is already in the tree
        self.assertEqual(tree.nearest_neighbor(self.n_2.q), self.n_2)

    def test_get_path(self):
        tree = self.build_tree()

        # error should occur if the path root node is not a part of the Tree
        orphan_node = Node(np.array([5, 5]), None)
        with self.assertRaisesRegex(ValueError, "Node is not in the tree"):
            tree.get_path(orphan_node)

        path = tree.get_path(self.n_3)
        self.assertListEqual(path, [self.n_3, self.n_1, self.root])

        path = tree.get_path(self.root)
        self.assertListEqual(path, [self.root])

    def test_sink_node(self):
        tree = Tree(self.root, is_sink=True)
        self.assertIsNotNone(tree.sink_node)

        # Sink node is not used in nearest neighbor
        with self.assertRaisesRegex(
            RuntimeError, "Tree contains no valid nodes for nearest neighbor"
        ):
            tree.nearest_neighbor(np.array([1]))

        # Sink node is considered a part of the tree
        sink_duplicate = Node(tree.sink_node.q, self.root)
        self.assertIn(sink_duplicate, tree)
        with self.assertRaisesRegex(ValueError, "already exists in the tree"):
            tree.add_node(sink_duplicate)

        # Sink node is not used in nearest neighbor, so make sure that calling
        # nearest neighbor with a q that matches the sink node's q returns the
        # nearest node that is NOT the sink node
        tree.add_node(self.n_1)
        nn = tree.nearest_neighbor(tree.sink_node.q)
        self.assertNotEqual(nn, tree.sink_node)
        self.assertEqual(nn, self.n_1)

        # Sink node should be ignored in get_path
        path = tree.get_path(tree.sink_node)
        self.assertEqual(len(path), 0)
        path = tree.get_path(self.n_1)
        self.assertListEqual(path, [self.n_1])


if __name__ == "__main__":
    unittest.main()

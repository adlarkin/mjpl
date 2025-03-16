import unittest

import numpy as np

from mj_maniPlan.node import Node
from mj_maniPlan.tree import Tree


class TestTree(unittest.TestCase):
    def build_tree(self):
        nodes = {self.n_0, self.n_1, self.n_2, self.n_3}
        for n in nodes:
            self.tree.add_node(n)
        self.assertSetEqual(self.tree.nodes, nodes)

    def setUp(self):
        """
        The nodes defined here will form the following tree:

                        n_2
                         ^
                         |
                        n_0 -> n_1 -> n_3
        """
        self.tree = Tree()
        self.n_0 = Node(np.array([0, 0]), None)
        self.n_1 = Node(np.array([1, 0]), self.n_0)
        self.n_2 = Node(np.array([0, 1]), self.n_0)
        self.n_3 = Node(np.array([2, 0]), self.n_1)

    def test_add_node(self):
        self.build_tree()

        # Ignore adding a node that's already in the tree
        num_nodes = len(self.tree.nodes)
        duplicate_node = Node(np.array([0, 0]), None)
        self.assertTrue(duplicate_node in self.tree.nodes)
        self.tree.add_node(duplicate_node)
        self.assertEqual(num_nodes, len(self.tree.nodes))

        # Add a node that's not already in the tree
        new_node = Node(np.array([-1, -1]), self.n_0)
        self.assertFalse(new_node in self.tree.nodes)
        self.tree.add_node(new_node)
        self.assertTrue(new_node in self.tree.nodes)

    def test_nearest_neighbor(self):
        q = np.array([2, 1])

        # error should occur if nearest_neighbor is called before any nodes have been added to the tree
        with self.assertRaisesRegex(ValueError, "call this method before adding"):
            self.tree.nearest_neighbor(q)

        self.build_tree()

        nn = self.tree.nearest_neighbor(q)
        self.assertEqual(nn, self.n_3)

        # check nearest neighbor for a q that is equidistant to multiple nodes in the tree
        self.assertTrue(
            self.tree.nearest_neighbor(np.array([1, 1])) in {self.n_1, self.n_2}
        )

        # check nearest neighbor for a q that is already in the tree
        self.assertEqual(self.tree.nearest_neighbor(self.n_2.q), self.n_2)

    def test_get_path(self):
        self.build_tree()

        # error should occur if the path root node is not a part of the Tree
        orphan_node = Node(np.array([5, 5]), None)
        with self.assertRaisesRegex(
            ValueError, "Called get_path starting from a node that is not in the tree"
        ):
            self.tree.get_path(orphan_node)

        path = self.tree.get_path(self.n_3)
        self.assertTrue(np.array_equal(path, [self.n_3.q, self.n_1.q, self.n_0.q]))

        path = self.tree.get_path(self.n_0)
        self.assertTrue(np.array_equal(path, [self.n_0.q]))


if __name__ == "__main__":
    unittest.main()

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Node:
    """Node object which can be used in a tree."""

    q: np.ndarray
    parent: "Node | None" = None

    def __hash__(self):
        """Hash based only on q to ensure uniqueness per tree."""
        # Convert q to a tuple for hashing
        return hash(tuple(self.q))

    def __eq__(self, other):
        """Nodes are equal if their q values are equal (ignores parent)."""
        if not isinstance(other, Node):
            return False
        return np.array_equal(self.q, other.q)


class Tree:
    """Tree of nodes."""

    def __init__(self):
        """Constructor."""
        self.nodes = set()
        self.q_to_node = {}

    def add_node(self, q: np.ndarray, parent: Node | None = None) -> Node:
        """Add a node to a tree.

        Args:
            q: The node's q value.
            parent: The node's parent.

        Return:
            The node from the tree.

        Notes:
            If there's already a node in the tree that has the specified `q`,
            a new node will not be added to the tree and this pre-existing
            node is what is returned.
        """
        q_tuple = tuple(q)

        if q_tuple in self.q_to_node:
            return self.q_to_node[q_tuple]

        new_node = Node(q, parent)
        self.nodes.add(new_node)
        self.q_to_node[q_tuple] = new_node
        return new_node

    def __contains__(self, node: Node) -> bool:
        """Check if a node is in the tree."""
        return node in self.nodes

    def nearest_neighbor(self, q: np.ndarray) -> Node:
        """Finds the nearest node in the tree to the given q.

        Args:
            q: The desired q.

        Returns: The node from the tree that's closest to `q`.
        """
        if not self.nodes:
            raise ValueError("Tree is empty, cannot find nearest neighbor.")

        q_tuple = tuple(q)
        if q_tuple in self.q_to_node:
            return self.q_to_node[q_tuple]
        return min(self.nodes, key=lambda node: np.linalg.norm(node.q - q))

    def get_path(self, node: Node) -> list[np.ndarray]:
        """Get the path of q's from the node to the root of the tree.

        Args:
            node: The node that defines the start of the path.

        Returns:
            A list of each node's q from `node` to the root of the tree.
        """
        if node not in self.nodes:
            raise ValueError("Node is not in the tree.")

        path = []
        curr_node = node
        while curr_node is not None:
            path.append(curr_node.q)
            curr_node = curr_node.parent
        return path

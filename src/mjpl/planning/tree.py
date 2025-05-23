from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Node:
    """Node object which can be used in a tree."""

    q: np.ndarray
    parent: Node | None = None

    def __hash__(self):
        """Hash based only on q to ensure uniqueness per tree."""
        return hash(self.q.tobytes())

    def __eq__(self, other):
        """Nodes are equal if their q values are equal (ignores parent)."""
        if not isinstance(other, Node):
            return False
        return np.array_equal(self.q, other.q)


class Tree:
    """Tree of nodes."""

    def __init__(self, root: Node):
        """Constructor.

        Args:
            root: The tree's root node, which should have no parent.
        """
        if root.parent:
            raise ValueError("The root node should have no parent.")
        self.nodes = {root}

    def add_node(self, node: Node):
        """Add a node to a tree.

        Args:
            node: The node to add.
        """
        if not node.parent:
            raise ValueError("Node does not have a parent.")
        if node in self.nodes:
            raise ValueError(f"A node with q={node.q} already exists in the tree.")
        if node.parent not in self.nodes:
            raise ValueError("Node's parent is not in the tree.")
        self.nodes.add(node)

    def __contains__(self, node: Node) -> bool:
        """Check if a node is in the tree."""
        return node in self.nodes

    def nearest_neighbor(self, q: np.ndarray) -> Node:
        """Finds the nearest node in the tree to the given q.

        Args:
            q: The desired q.

        Returns: The node from the tree that's closest to `q`.
        """
        closest_node = min(self.nodes, key=lambda node: np.linalg.norm(node.q - q))
        return closest_node

    def get_path(self, node: Node) -> list[Node]:
        """Get the path from the given node to the tree's root.

        Args:
            node: The node that defines the start of the path.

        Returns:
            A list of nodes to the root, starting from `node`.
        """
        if node not in self.nodes:
            raise ValueError("Node is not in the tree.")

        path = []
        curr_node = node
        while curr_node is not None:
            path.append(curr_node)
            curr_node = curr_node.parent
        return path

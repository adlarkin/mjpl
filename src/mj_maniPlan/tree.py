import numpy as np


class Node:
    def __init__(self, q: np.ndarray, parent) -> None:
        self.q = q
        self.parent = parent

    def __hash__(self) -> int:
        return hash(self.q.tobytes())

    # Define node equality as having the same jont config value.
    # TODO: update this? Might want to check for the same parent.
    # There is also the chance that two nodes can have the same config value,
    # but belong to different trees (maybe enforcing a parent check will help resolve this).
    def __eq__(self, other) -> bool:
        return np.array_equal(self.q, other.q)


class Tree:
    def __init__(self) -> None:
        # For now, the tree is represented as a set of unique nodes.
        # TODO: use something like a kd-tree to improve nearest neighbor lookup times?
        # https://github.com/adlarkin/mj_maniPlan/issues/19
        self.nodes = set()

    def add_node(self, node: Node):
        self.nodes.add(node)

    def nearest_neighbor(self, q: np.ndarray) -> Node:
        closest_node = None
        min_dist = np.inf
        for n in self.nodes:
            if np.array_equal(n.q, q):
                return n
            neighboring_dist = np.linalg.norm(q - n.q)
            if neighboring_dist < min_dist:
                closest_node = n
                min_dist = neighboring_dist
        if not closest_node:
            raise ValueError(
                f"No nearest neighbor found for {q}. Did you call this method before adding any nodes to the tree?"
            )
        return closest_node

    def get_path(self, node: Node) -> list[np.ndarray]:
        if node not in self.nodes:
            raise ValueError(
                "Called get_path starting from a node that is not in the tree."
            )
        path = []
        curr_node = node
        while curr_node is not None:
            path.append(curr_node.q)
            curr_node = curr_node.parent
        return path

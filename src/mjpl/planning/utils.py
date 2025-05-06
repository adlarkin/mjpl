import numpy as np

from .. import utils
from ..constraint.constraint_interface import Constraint
from ..constraint.utils import apply_constraints
from .tree import Node, Tree


def _extend(
    q_target: np.ndarray,
    tree: Tree,
    start_node: Node,
    eps: float,
    constraints: list[Constraint],
) -> Node | None:
    """Extend a node in a tree towards a target configuration.

    Args:
        q_target: The target configuration.
        tree: The tree with a node to extend towards `q_target`.
        start_node: The node in `tree` to extend towards `q_target`.
        eps: The maximum distance `start_node` will extend towards `q_target`.
        constraints: The constraints the extended configuration must obey.

    Returns:
        The node that was the result of extending `start_node` towards `q_target`,
        or None if extension wasn't possible. This node also belongs to `tree`.
    """
    if np.array_equal(start_node.q, q_target):
        return start_node
    q_extend = utils.step(start_node.q, q_target, eps)
    q_constrained = apply_constraints(start_node.q, q_extend, constraints)
    if q_constrained is not None:
        extended_node = Node(q_constrained, start_node)
        # Applying constraints can result in configurations that are already in the tree.
        # TODO: figure out if I need this, or if it's a bug
        # If the node exists, should I return the existing node? (need to get the parent right)
        if extended_node not in tree:
            tree.add_node(extended_node)
            return extended_node
    return None


def _connect(
    q_target: np.ndarray,
    tree: Tree,
    eps: float,
    max_connection_distance: float,
    constraints: list[Constraint],
) -> Node:
    """Attempt to connect a node in a tree to a target configuration.

    Args:
        q_target: The target configuration.
        tree: The tree with a node that serves as the basis of the connection
            to `q_target`.
        eps: The maximum distance between nodes added to `tree`. If the
            distance between the start node in `tree` and `q_target` is greater
            than `eps`, multiple nodes will be added to `tree`.
        max_connection_distance: The maximum distance to cover before terminating
            the connect operation.
        constraints: The constraints configurations must obey.

    Returns:
        The node that is the result of connecting a node from `tree` towards
        `q_target`. This node also belongs to `tree`.
    """
    nearest_node = tree.nearest_neighbor(q_target)
    q_old = nearest_node.q
    total_distance = 0.0
    while not np.array_equal(nearest_node.q, q_target):
        max_eps = min(eps, max_connection_distance - total_distance)
        next_node = _extend(q_target, tree, nearest_node, max_eps, constraints)
        # Terminate if extension failed, or if extension is not making progress
        # towards q_target because of the constraints.
        if not next_node or _deviates_from_target(q_target, next_node.q, q_old):
            break
        q_old = next_node.q
        nearest_node = next_node
        total_distance += max_eps
        if total_distance >= max_connection_distance:
            break
    return nearest_node


def _deviates_from_target(
    target: np.ndarray, curr: np.ndarray, prev: np.ndarray
) -> bool:
    """Check if a vector deviates from a target w.r.t. another vector.

    Args:
        target: The target vector.
        curr: The vector to evaluate.
        prev: The previous vector to compare against.

    Returns:
        True if `curr` is further from `target` than `prev`. False otherwise.
    """
    return np.linalg.norm(target - curr) > np.linalg.norm(target - prev)


def _combine_paths(
    start_tree: Tree,
    start_tree_node: Node,
    goal_tree: Tree,
    goal_tree_node: Node,
) -> list[np.ndarray]:
    """Combine paths from a start and goal tree.

    Args:
        start_tree: The tree whose root is the start of the combined path.
        start_tree_node: The node in `start_tree` that marks the end of the
            path that begins at the root of `start_tree`.
        goal_tree: The tree whose root is the end of the combined path.
        goal_tree_node: The node in `goal_tree` that marks the beginning of
            the path that ends at the root of `goal_tree`.

    Returns:
        A path that starts at the root of `start_tree` and ends at `goal_tree`,
        with a connecting edge between `start_tree_node` and `goal_tree_node`.
    """
    # The path generated from start_tree ends at q_init, but we want it to
    # start at q_init. So we must reverse it.
    path_start = [n.q for n in start_tree.get_path(start_tree_node)]
    path_start.reverse()
    # The path generated from goal_tree ends at q_goal, which is what we want.
    path_end = [n.q for n in goal_tree.get_path(goal_tree_node)]
    # The last value in path_start might be the same as the first value in
    # path_end. If this is the case, remove the duplicate value.
    if np.array_equal(path_start[-1], path_end[0]):
        path_start.pop()
    return path_start + path_end

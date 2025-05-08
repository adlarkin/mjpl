import numpy as np

from .. import utils
from ..constraint.constraint_interface import Constraint
from ..constraint.utils import apply_constraints
from .tree import Node, Tree


def smooth_path(
    waypoints: list[np.ndarray],
    constraints: list[Constraint],
    eps: float = 0.05,
    num_tries: int = 100,
    seed: int | None = None,
    sparse: bool = False,
) -> list[np.ndarray]:
    smoothed_path = waypoints
    rng = np.random.default_rng(seed=seed)
    for _ in range(num_tries):
        # Randomly select two waypoints to shortcut.
        start = rng.integers(0, len(smoothed_path) - 1)
        end = rng.integers(start + 1, len(smoothed_path))
        assert start < end
        assert start >= 0 and start < len(smoothed_path) - 1
        assert end < len(smoothed_path)

        # See if the randomly selected waypoints can be connected without violating
        # constraints.
        node = Node(smoothed_path[start])
        tree = Tree(node)
        q_reached = _constrained_extend(
            smoothed_path[end], node, tree, eps, constraints
        )
        if np.array_equal(q_reached, smoothed_path[end]):
            # Form a direct connection between the two waypoints if it shortens path
            # length. This check must be performed since constraints project
            # configurations in an arbitrary way.
            end_node = tree.nearest_neighbor(q_reached)
            shortcut_path_segment = [n.q for n in tree.get_path(end_node)]
            shortcut_length = utils.path_length(shortcut_path_segment)
            original_length = utils.path_length(smoothed_path[start : end + 1])
            if shortcut_length < original_length:
                # tree.get_path gives order from the specified node to the tree's root,
                # so we must reverse it to get ordering starting from the root.
                shortcut_path_segment.reverse()
                if sparse:
                    smoothed_path = smoothed_path[: start + 1] + smoothed_path[end:]
                else:
                    smoothed_path = (
                        smoothed_path[:start]
                        + shortcut_path_segment[:-1]
                        + smoothed_path[end:]
                    )

        if len(smoothed_path) == 2:
            # The first and last waypoints can be directly connected.
            break
    return smoothed_path


def _constrained_extend(
    q_target: np.ndarray,
    nearest_node: Node,
    tree: Tree,
    eps: float,
    constraints: list[Constraint],
) -> np.ndarray:
    q = nearest_node.q
    q_old = nearest_node.q
    closest_node = nearest_node

    while True:
        if np.array_equal(q_target, q):
            return q

        q = utils.step(q, q_target, eps)
        q = apply_constraints(q_old, q, constraints)

        # Terminate if:
        # - Applying constraints failed
        # - The configuration is the same as q_old after applying constraints
        # - Applying constraints gives a configuration that deviates from q_target
        if (
            q is None
            or np.linalg.norm(q - q_old) < 1e-8
            or np.linalg.norm(q_target - q) > np.linalg.norm(q_target - q_old)
        ):
            return q_old

        closest_node = Node(q, closest_node)
        tree.add_node(closest_node)
        q_old = q


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

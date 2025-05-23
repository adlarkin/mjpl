import numpy as np

from ..constraint.collision_constraint import CollisionConstraint
from ..constraint.constraint_interface import Constraint
from ..constraint.utils import apply_constraints
from .tree import Node, Tree


def smooth_path(
    waypoints: list[np.ndarray],
    constraints: list[Constraint],
    collision_interval_check: tuple[float, CollisionConstraint] | None = None,
    eps: float = 0.05,
    num_tries: int = 100,
    seed: int | None = None,
    sparse: bool = False,
) -> list[np.ndarray]:
    """Smooth a path subject to constraints.

    This is based on algorithm 3, described here:
    https://personalrobotics.cs.washington.edu/publications/berenson2009cbirrt.pdf

    Args:
        waypoints: The waypoints that form the path to be smoothed.
        constraints: The constraints the smoothed path must obey.
        collision_interval_check: A tuple that defines the step distance and
            CollisionConstraint that are used to check if the interval between two
            configurations obeys the CollisionConstraint. Interval checking is disabled
            if this is None.
        eps: The step size that is used for checking constraints when attempting to
            smooth `waypoints`. If `sparse` is False, this is the maximum distance
            between waypoints that are added to the smoothed path.
        num_tries: The number of times to randomly select two waypoints and attempt
            smoothing between them.
        seed: The seed for the underlying random number generator.
        sparse: If True, a "sparse" path is formed. A sparse path only keeps waypoints
            from the original waypoints list. If False, constrained waypoints are added
            in between smoothed waypoints to ensure that the smoothed path has consecutive
            waypoints that are no further than `eps` apart.

    Returns:
        A smoothed path based on `waypoints` that obeys `constraints`.
    """
    if not waypoints:
        raise ValueError("`waypoints` cannot be empty.")
    if eps <= 0.0:
        raise ValueError("`eps` must be > 0.")
    if num_tries <= 0:
        raise ValueError("`num_tries` must be > 0.")

    smoothed_path = waypoints
    rng = np.random.default_rng(seed=seed)
    for _ in range(num_tries):
        # Randomly select two waypoints to shortcut.
        start = rng.integers(0, len(smoothed_path) - 1)
        end = rng.integers(start + 1, len(smoothed_path))

        # See if the randomly selected waypoints can be connected without violating
        # constraints.
        tree = Tree(Node(smoothed_path[start]))
        q_reached = _constrained_extend(
            smoothed_path[end], tree, eps, constraints, collision_interval_check
        )
        if not np.array_equal(q_reached, smoothed_path[end]):
            continue

        # Form a direct connection between the two waypoints if it shortens path
        # length. This check must be performed since constraints project
        # configurations in an arbitrary way.
        end_node = tree.nearest_neighbor(q_reached)
        shortcut_path_segment = [n.q for n in tree.get_path(end_node)]
        shortcut_length = path_length(shortcut_path_segment)
        original_length = path_length(smoothed_path[start : end + 1])
        if shortcut_length < original_length:
            if sparse:
                smoothed_path = smoothed_path[: start + 1] + smoothed_path[end:]
            else:
                # tree.get_path gives order from the specified node to the tree's root,
                # so we must reverse it to get ordering starting from the root.
                shortcut_path_segment.reverse()
                smoothed_path = (
                    smoothed_path[:start]
                    + shortcut_path_segment[:-1]
                    + smoothed_path[end:]
                )

    return smoothed_path


def path_length(waypoints: list[np.ndarray]) -> float:
    """Compute the path length in configuration space.

    Args:
        waypoints: A list of waypoints that form the path.

    Returns:
        The length of the waypoint list in configuration space.
    """
    path = np.array(waypoints)
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(segment_lengths)


def _constrained_extend(
    q_target: np.ndarray,
    tree: Tree,
    eps: float,
    constraints: list[Constraint],
    collision_interval_check: tuple[float, CollisionConstraint] | None = None,
    equality_threshold: float = 1e-8,
) -> np.ndarray:
    """Extend a tree towards a target configuration, subject to constraints.

    This is based on algorithm 2, described here:
    https://personalrobotics.cs.washington.edu/publications/berenson2009cbirrt.pdf

    Args:
        q_target: The target configuration.
        tree: The tree to extend towards `q_target`.
        eps: The maximum distance allowed between nodes in `tree`.
        constraints: The constraints nodes in `tree` must obey.
        collision_interval_check: A tuple that defines the step distance and
            CollisionConstraint that are used to check if the interval between two
            configurations obeys the CollisionConstraint. Interval checking is disabled
            if this is None.
        equality_threshold: Configuration distance threshold for determining whether or
            not a constrained configuration is equivalent to its previous configuration.
            Used as termination criteria for handling things like local minima in
            constraint functions.

    Returns:
        The configuration that `tree` was able to reach.
    """
    closest_node = tree.nearest_neighbor(q_target)
    q = closest_node.q
    q_old = closest_node.q

    while True:
        if np.array_equal(q_target, q):
            return q

        q = _step(q, q_target, eps)
        q = apply_constraints(q_old, q, constraints)

        # Terminate if:
        # - Applying constraints failed
        # - The configuration is the same as q_old after applying constraints
        # - Applying constraints gives a configuration that deviates from q_target
        # - The interval between q_old and the configuration violates collisions
        if (
            q is None
            or np.linalg.norm(q - q_old) < equality_threshold
            or np.linalg.norm(q_target - q) > np.linalg.norm(q_target - q_old)
            or (
                collision_interval_check is not None
                and not _valid_collision_interval(q_old, q, *collision_interval_check)
            )
        ):
            return q_old

        closest_node = Node(q, closest_node)
        tree.add_node(closest_node)
        q_old = q


def _step(start: np.ndarray, target: np.ndarray, max_step_dist: float) -> np.ndarray:
    """Step a vector towards a target.

    Args:
        start: The start vector.
        target: The target.
        max_step_dist: Maximum amount to step towards `target`.

    Return:
        A vector that has taken a step towards `target` from `start`.
    """
    if max_step_dist <= 0.0:
        raise ValueError("`max_step_dist` must be > 0.0")
    if np.array_equal(start, target):
        return start.copy()
    direction = target - start
    magnitude = np.linalg.norm(direction)
    unit_vec = direction / magnitude
    return start + (unit_vec * min(max_step_dist, magnitude))


def _valid_collision_interval(
    start: np.ndarray,
    end: np.ndarray,
    step_dist: float,
    constraint: CollisionConstraint,
) -> bool:
    """Check if configurations at a discrete step over an interval (excluding the
    start/end points) obey a CollisionConstraint.

    Args:
        start: The configuration that marks the start of the interval.
        end: The configuration that marks the end of the interval.
        step_dist: The distance to step towards `end` from `start`.
        constraint: The CollisionConstraint that configurations in the interval must obey.

    Returns:
        True if the configurations between `start` and `end` obey `constraint`.
        False otherwise.
    """
    if step_dist <= 0.0:
        raise ValueError("`step_dist` must be > 0")

    waypoints = [start]
    while not np.array_equal(waypoints[-1], end):
        waypoints.append(_step(waypoints[-1], end, step_dist))
    # Ignore the start/end of the interval (only check intermediate configurations).
    waypoints = waypoints[1:-1]

    return all(constraint.valid_config(wp) for wp in waypoints)


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

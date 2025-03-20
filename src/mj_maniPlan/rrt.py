import time
from dataclasses import dataclass

import mujoco
import numpy as np
from mink.lie.se3 import SE3

from . import utils
from .collision_ruleset import CollisionRuleset
from .inverse_kinematics.ik_solver import IKSolver
from .inverse_kinematics.mink_ik_solver import MinkIKSolver
from .joint_group import JointGroup
from .tree import Node, Tree


@dataclass
class RRTOptions:
    # The joints to plan for.
    jg: JointGroup
    # The collision rules to enforce during planning.
    cr: CollisionRuleset
    # Maximum planning time, in seconds.
    # If this value is <= 0, the planner will run until a solution is found.
    # A value <= 0 may lead to infinite run time, since sampling-based planners are probabilistically complete!
    max_planning_time: float = 10.0
    # The RRT "growth factor".
    # This is the maximum distance between nodes in the tree.
    # This number should be > 0.
    epsilon: float = 0.05
    # Seed used for the underlying sampler in the planner.
    # `None` means the algorithm is nondeterministic.
    seed: int | None = None
    # How often to sample the goal state when building the tree.
    # This should be a value within [0.0, 1.0].
    goal_biasing_probability: float = 0.05
    # The maximum distance for extending a tree using CONNECT.
    max_connection_distance: float = np.inf


# Implementation of Bidirectional RRT-Connect:
# https://www.cs.cmu.edu/afs/cs/academic/class/15494-s14/readings/kuffner_icra2000.pdf
#
# The reference above runs EXTEND on one tree and CONNECT on the other, swapping trees every iteration.
# This implementation runs CONNECT on both trees, removing the need for tree swapping.
class RRT:
    def __init__(self, options: RRTOptions) -> None:
        # During planning, we need to use MjData for things like forward kinematics
        # to confirm that sampled configurations are valid.
        self.data = mujoco.MjData(options.jg.model)

        self.options = options
        self.rng = np.random.default_rng(seed=options.seed)

    def plan_to_pose(
        self,
        pose: SE3,
        site: str,
        q_init_world: np.ndarray,
        solver: IKSolver | None = None,
    ) -> list[np.ndarray]:
        if solver is None:
            solver = MinkIKSolver(
                model=self.options.jg.model,
                jg=self.options.jg,
                cr=self.options.cr,
                q_init_guess=q_init_world,
                seed=self.options.seed,
                max_attempts=5,
            )
        q_goal = solver.solve_ik(pose, site)
        if q_goal is None:
            print("Unable to find a configuration for the target pose.")
            return []

        return self.plan_to_config(q_init_world, q_goal[self.options.jg.qpos_addrs])

    def plan_to_config(
        self, q_init_world: np.ndarray, q_goal: np.ndarray
    ) -> list[np.ndarray]:
        assert q_init_world.size == self.data.qpos.size
        assert q_goal.size == len(self.options.jg.joint_ids)

        self.data.qpos = q_init_world
        q_init = self.options.jg.qpos(self.data)
        if not utils.is_valid_config(
            q_init, self.options.jg, self.data, self.options.cr
        ):
            print("q_init is not a valid configuration")
            return []
        if not utils.is_valid_config(
            q_goal, self.options.jg, self.data, self.options.cr
        ):
            print("q_goal is not a valid configuration")
            return []

        # Is there a direct connection to q_goal from q_init?
        if utils.configuration_distance(q_init, q_goal) < self.options.epsilon:
            return [q_init, q_goal]

        start_tree = Tree(Node(q_init))
        goal_tree = Tree(Node(q_goal))

        max_planning_time = self.options.max_planning_time
        if max_planning_time <= 0:
            max_planning_time = float("inf")

        start_time = time.time()
        while time.time() - start_time < max_planning_time:
            if self.rng.random() <= self.options.goal_biasing_probability:
                q_rand = q_goal
            else:
                q_rand = self.options.jg.random_config(self.rng)

            new_start_tree_node = self._connect(q_rand, start_tree)
            new_goal_tree_node = self._connect(new_start_tree_node.q, goal_tree)
            if (
                utils.configuration_distance(
                    new_start_tree_node.q, new_goal_tree_node.q
                )
                < self.options.epsilon
            ):
                return self._combine_paths(
                    start_tree, new_start_tree_node, goal_tree, new_goal_tree_node
                )

            if not np.array_equal(new_start_tree_node.q, q_rand):
                new_goal_tree_node = self._connect(q_rand, goal_tree)
                new_start_tree_node = self._connect(new_goal_tree_node.q, start_tree)
                if (
                    utils.configuration_distance(
                        new_start_tree_node.q, new_goal_tree_node.q
                    )
                    < self.options.epsilon
                ):
                    return self._combine_paths(
                        start_tree, new_start_tree_node, goal_tree, new_goal_tree_node
                    )

        return []

    def _extend(
        self,
        q_target: np.ndarray,
        tree: Tree,
        start_node: Node | None = None,
        eps: float | None = None,
    ) -> Node | None:
        """Extend a node in a tree towards a target configuration.

        Args:
            q_target: The target configuration.
            tree: The tree with a node to extend towards `q_target`.
            start_node: The node in `tree` to extend towards `q_target`.
                If this isn't defined, the node closest to `q_target` in
                `tree` will be used.
            eps: The maximum distance `start_node` will extend towards
                `q_target`. If this isn't defined, the planner's epsilon
                parameter will be used (see `RRTOptions.epsilon`).

        Returns:
            The node that was the result of extending `start_node` towards
            `q_target`, or None if extension wasn't possible. This node also
            belongs to `tree`.
        """
        eps = eps or self.options.epsilon
        start_node = start_node or tree.nearest_neighbor(q_target)
        if np.array_equal(start_node.q, q_target):
            return start_node
        q_extend = utils.step(start_node.q, q_target, eps)
        if utils.is_valid_config(q_extend, self.options.jg, self.data, self.options.cr):
            extended_node = Node(q_extend, start_node)
            tree.add_node(extended_node)
            return extended_node
        return None

    def _connect(
        self,
        q_target: np.ndarray,
        tree: Tree,
        eps: float | None = None,
        max_connection_distance: float | None = None,
    ) -> Node:
        """Attempt to connect a node in a tree to a target configuration.

        Args:
            q_target: The target configuration.
            tree: The tree with a node that serves as the basis of the connection
                to `q_target`.
            eps: The maximum distance between nodes added to `tree`. If the
                distance between the start node in `tree` and `q_target` is greater
                than `eps`, multiple nodes will be added to `tree`. If this isn't
                defined, the planner's epsilon parameter will be used
                (see `RRTOptions.epsilon`).
            max_connection_distance: The maximum distance to cover before
                terminating the connect operation. If this isn't defined, the
                planner's max_connection_distance parameter will be used (see
                `RRTOptions.max_connection_distance`).

        Returns:
            The node that is the result of connecting a node from `tree` towards
            `q_target`. This node also belongs to `tree`.
        """
        eps = eps or self.options.epsilon
        max_connection_distance = (
            max_connection_distance or self.options.max_connection_distance
        )

        nearest_node = tree.nearest_neighbor(q_target)
        total_distance = 0.0
        while not np.array_equal(nearest_node.q, q_target):
            max_eps = min(eps, max_connection_distance - total_distance)
            next_node = self._extend(
                q_target, tree, start_node=nearest_node, eps=max_eps
            )
            if not next_node:
                break
            nearest_node = next_node
            total_distance += max_eps
            if total_distance >= max_connection_distance:
                break
        return nearest_node

    def _combine_paths(
        self,
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
        # The path generated from start_tree ends at q_init, but we want it to start at q_init. So we must reverse it.
        path_start = [n.q for n in start_tree.get_path(start_tree_node)]
        path_start.reverse()
        # The path generated from goal_tree ends at q_goal, which is what we want.
        path_end = [n.q for n in goal_tree.get_path(goal_tree_node)]
        # The last value in path_start might be the same as the first value in path_end.
        # If this is the case, remove the duplicate value.
        if np.array_equal(path_start[-1], path_end[0]):
            path_start.pop()
        return path_start + path_end

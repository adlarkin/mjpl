import time
from dataclasses import dataclass

import mujoco
import numpy as np

from . import utils
from .collision_ruleset import CollisionRuleset
from .inverse_kinematics import IKOptions, solve_ik
from .joint_group import JointGroup


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
        self.nodes = set()

    def add_node(self, node: Node):
        self.nodes.add(node)

    def nearest_neighbor(self, q: np.ndarray) -> Node:
        closest_node = None
        min_dist = np.inf
        for n in self.nodes:
            if np.array_equal(n.q, q):
                return n
            neighboring_dist = utils.configuration_distance(n.q, q)
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
        q_init_world: np.ndarray,
        site: str,
        pos: np.ndarray,
        rot: np.ndarray,
        ik_opts: IKOptions,
    ) -> list[np.ndarray]:
        if ik_opts.jg != self.options.jg:
            raise RuntimeError("Joint groups must be the same in IK and RRT options.")
        if ik_opts.cr != self.options.cr:
            raise RuntimeError(
                "Collision rulesets must be the same in IK and RRT options."
            )

        q_goal = solve_ik(site, q_init_world, pos, rot, ik_opts)
        if q_goal is None:
            print("Unable to find a configuration for the target pose.")
            return []

        # The IK solver gives a full world configuration, but we only care about joints in the JointGroup for planning.
        self.data.qpos = q_goal.copy()
        q_goal = self.options.jg.qpos(self.data)
        if not utils.is_valid_config(
            q_goal, self.options.jg, self.data, self.options.cr
        ):
            print(
                "The configuration that satisfies the target pose violates allowed collisions."
            )
            return []

        return self.plan_to_config(q_init_world, q_goal)

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

        start_tree = Tree()
        start_tree.add_node(Node(q_init, None))
        goal_tree = Tree()
        goal_tree.add_node(Node(q_goal, None))

        max_planning_time = self.options.max_planning_time
        if max_planning_time <= 0:
            max_planning_time = float("inf")

        start_time = time.time()
        while time.time() - start_time < max_planning_time:
            if self.rng.random() <= self.options.goal_biasing_probability:
                q_rand = q_goal
            else:
                q_rand = self.options.jg.random_config(self.rng)

            new_start_tree_node = self.connect(q_rand, start_tree)
            new_goal_tree_node = self.connect(new_start_tree_node.q, goal_tree)
            if (
                utils.configuration_distance(
                    new_start_tree_node.q, new_goal_tree_node.q
                )
                < self.options.epsilon
            ):
                return self.get_path(
                    start_tree, new_start_tree_node, goal_tree, new_goal_tree_node
                )

            if not np.array_equal(new_start_tree_node.q, q_rand):
                new_goal_tree_node = self.connect(q_rand, goal_tree)
                new_start_tree_node = self.connect(new_goal_tree_node.q, start_tree)
                if (
                    utils.configuration_distance(
                        new_start_tree_node.q, new_goal_tree_node.q
                    )
                    < self.options.epsilon
                ):
                    return self.get_path(
                        start_tree, new_start_tree_node, goal_tree, new_goal_tree_node
                    )

        return []

    def extend(
        self,
        q: np.ndarray,
        tree: Tree,
        nearest_node: Node | None = None,
        eps: float | None = None,
    ) -> Node | None:
        eps = eps or self.options.epsilon
        nearest_node = nearest_node or tree.nearest_neighbor(q)
        if np.array_equal(nearest_node.q, q):
            return nearest_node
        q_extend = q.copy()
        q_dist = utils.configuration_distance(nearest_node.q, q)
        if q_dist > eps:
            q_increment = eps * ((q - nearest_node.q) / q_dist)
            q_extend = nearest_node.q + q_increment
        if utils.is_valid_config(q_extend, self.options.jg, self.data, self.options.cr):
            node_extend = Node(q_extend, nearest_node)
            tree.add_node(node_extend)
            return node_extend
        return None

    def connect(
        self,
        q: np.ndarray,
        tree: Tree,
        eps: float | None = None,
        max_connection_distance: float | None = None,
    ) -> Node:
        eps = eps or self.options.epsilon
        max_connection_distance = (
            max_connection_distance or self.options.max_connection_distance
        )

        nearest_node = tree.nearest_neighbor(q)
        total_distance = 0.0
        while not np.array_equal(q, nearest_node.q):
            max_eps = min(eps, max_connection_distance - total_distance)
            next_node = self.extend(q, tree, nearest_node=nearest_node, eps=max_eps)
            if not next_node:
                break
            nearest_node = next_node
            total_distance += max_eps
            if total_distance >= max_connection_distance:
                break
        return nearest_node

    def get_path(
        self,
        start_tree: Tree,
        start_tree_node: Node,
        goal_tree: Tree,
        goal_tree_node: Node,
    ) -> list[np.ndarray]:
        # The path generated from start_tree ends at q_init, but we want it to start at q_init. So we must reverse it.
        path_start = start_tree.get_path(start_tree_node)
        path_start.reverse()
        # The path generated from goal_tree ends at q_goal, which is what we want.
        path_end = goal_tree.get_path(goal_tree_node)
        # The last value in path_start might be the same as the first value in path_end.
        # If this is the case, remove the duplicate value.
        if np.array_equal(path_start[-1], path_end[0]):
            path_start.pop()
        return path_start + path_end

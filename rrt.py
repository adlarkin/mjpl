import copy
import mujoco
import numpy as np
import time

from dataclasses import dataclass


def configuration_distance(q_from, q_to):
    return np.linalg.norm(q_to - q_from)

def joint_names_to_qpos_addrs(joint_names, model: mujoco.MjModel):
    return np.array([ model.joint(name).qposadr.item() for name in joint_names ])

def joint_limits(joint_names, model: mujoco.MjModel):
    lower_limits = np.array([ model.joint(name).range[0] for name in joint_names ])
    upper_limits = np.array([ model.joint(name).range[1] for name in joint_names ])
    return lower_limits, upper_limits

def random_config(rng: np.random.Generator, lower_limits, upper_limits):
    return rng.uniform(low=lower_limits, high=upper_limits)

# NOTE: this will modify `data` in-place.
def is_valid_config(q, lower_limits, upper_limits, qpos_addrs, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    # Check joint limits.
    if not ((q >= lower_limits) & (q <= upper_limits)).all():
        return False

    # Check for collisions.
    # We have to run FK once data.qpos is updated before running the collision checker.
    # TODO: enforce padding on the collision check? Not sure how to do this in MuJoCo yet
    data.qpos[qpos_addrs] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    return not data.ncon

# NOTE: this will modify `data` in-place, since it calls is_valid_config internally.
def random_valid_config(rng: np.random.Generator, lower_limits, upper_limits, joint_qpos_addrs, model: mujoco.MjModel, data: mujoco.MjData):
    q_rand = random_config(rng, lower_limits, upper_limits)
    while not is_valid_config(q_rand, lower_limits, upper_limits, joint_qpos_addrs, model, data):
        q_rand = random_config(rng, lower_limits, upper_limits)
    return q_rand


class Node:
    def __init__(self, q, parent) -> None:
        self.q = np.array(q)
        self.parent = parent

    def __hash__(self) -> int:
        return hash(self.q.tobytes())

    def __eq__(self, other) -> bool:
        return np.array_equal(self.q, other.q)


class Tree:
    def __init__(self) -> None:
        # For now, the tree is represented as a set of unique nodes.
        # TODO: use something like a kd-tree to improve nearest neighbor lookup times?
        self.nodes = set()

        # The node that defines the root of a path in the tree.
        # This should be set via set_path_root before calling get_path.
        self.path_root = None

    def add_node(self, node: Node):
        self.nodes.add(node)

    def nearest_neighbor(self, q) -> Node:
        closest_node = None
        min_dist = np.inf
        for n in self.nodes:
            if (n.q == q).all():
                return n
            neighboring_dist = configuration_distance(n.q, q)
            if neighboring_dist < min_dist:
                closest_node = n
                min_dist = neighboring_dist
        if not closest_node:
            raise ValueError(f"No nearest neighbor found for {q}. Did you call this method before adding any nodes to the tree?")
        return closest_node

    def set_path_root(self, node: Node):
        self.path_root = node

    def get_path(self) -> list[Node]:
        if not self.path_root:
            raise ValueError("The path root node has not been set. Did you forget to call set_path_root?")
        path = []
        curr_node = self.path_root
        while curr_node.parent is not None:
            path.append(curr_node.q)
            curr_node = curr_node.parent
        return path


@dataclass
class RRTOptions:
    # The joints to plan for.
    joint_names: list[str]
    # Maximum planning time, in seconds.
    # If this value is <= 0, the planner will run until a solution is found.
    # A value <= 0 may lead to infinite run time, since sampling-based planners are probabilistically complete!
    max_planning_time: float
    # The RRT "growth factor".
    # This is the maximum distance between nodes in the tree.
    # This number should be > 0.
    epsilon: float
    # Random number generator that's used for sampling joint configurations.
    rng: np.random.Generator
    # How often to sample the goal state when building the tree.
    # This should be a value within [0.0, 1.0].
    goal_biasing_probability: float = 0.05


# Implementation of Bidirectional RRT-Connect:
# https://www.cs.cmu.edu/afs/cs/academic/class/15494-s14/readings/kuffner_icra2000.pdf
class RRT:
    def __init__(self, options: RRTOptions, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # The model should be read-only, so we don't need to make a deep copy of it.
        self.model = model

        # We need the current state of the scene for collision checking, but we
        # also need to check joint configurations via FK to ensure that newly
        # sampled states are valid. 
        self.data = copy.deepcopy(data)

        self.options = options
        self.joint_qpos_addrs = joint_names_to_qpos_addrs(options.joint_names, self.model)
        self.joint_limits_lower, self.joint_limits_upper = joint_limits(options.joint_names, self.model)

    def plan(self, q_goal):
        # Use MjData's current joint configuration as q_init.
        q_init = self.data.qpos[self.joint_qpos_addrs]
        if not is_valid_config(q_init, self.joint_limits_lower, self.joint_limits_upper, self.joint_qpos_addrs, self.model, self.data):
            print("q_init is not a valid configuration")
            return []
        extend_tree = (Tree(), q_init)
        extend_tree[0].add_node(Node(q_init, None))

        if not is_valid_config(q_goal, self.joint_limits_lower, self.joint_limits_upper, self.joint_qpos_addrs, self.model, self.data):
            print("q_goal is not a valid configuration")
            return []
        connect_tree = (Tree(), q_goal)
        connect_tree[0].add_node(Node(q_goal, None))

        max_planning_time = self.options.max_planning_time
        if max_planning_time <= 0:
            max_planning_time = float('inf')

        # Is there a direct connection to q_goal from q_init?
        solution_found = False
        if configuration_distance(q_init, q_goal) <= self.options.epsilon:
            solution_found = True

        start_time = time.time()
        while (not solution_found and time.time() - start_time < max_planning_time):
            if self.options.rng.random() <= self.options.goal_biasing_probability:
                q_rand = connect_tree[1]
            else:
                q_rand = random_config(self.options.rng, self.joint_limits_lower, self.joint_limits_upper)
            extended_node = self.extend(q_rand, extend_tree[0])
            if extended_node:
                connected_node = self.connect(extended_node.q, connect_tree[0])
                # If extended_node and connected_node are within epsilon, we can connect the two trees.
                if connected_node and (configuration_distance(extended_node.q, connected_node.q) < self.options.epsilon):
                    extend_tree[0].set_path_root(extended_node)
                    connect_tree[0].set_path_root(connected_node)
                    solution_found = True

            # Swap trees for the next iteration.
            extend_tree, connect_tree = connect_tree, extend_tree

        print(f"Solution found: {solution_found}, time taken: {time.time() - start_time}")
        if solution_found:
            # The original extend_tree and connect_tree may have been swapped, so check which ones hold q_init and q_goal.
            if extend_tree[1] is q_goal:
                extend_tree, connect_tree = connect_tree, extend_tree
            return self.get_path(extend_tree[0], connect_tree[0])
        return []

    def extend(self, q, tree: Tree) -> Node | None:
        node_near = tree.nearest_neighbor(q)
        q_extend = q
        q_dist = configuration_distance(node_near.q, q)
        if q_dist > self.options.epsilon:
            q_increment = self.options.epsilon * ((q - node_near.q) / q_dist)
            q_extend = node_near.q + q_increment
        if is_valid_config(q_extend, self.joint_limits_lower, self.joint_limits_upper, self.joint_qpos_addrs, self.model, self.data):
            node_extend = Node(q_extend, node_near)
            tree.add_node(node_extend)
            return node_extend
        return None

    def connect(self, q, tree: Tree) -> Node | None:
        last_extended_node = None
        connection_dist_remaining = float('inf')
        while connection_dist_remaining > self.options.epsilon:
            next_node = self.extend(q, tree)
            if not next_node or (next_node.q == q).all():
                break
            last_extended_node = next_node
            connection_dist_remaining = configuration_distance(last_extended_node.q, q)
        return last_extended_node

    def get_path(self, start_tree: Tree, goal_tree: Tree) -> list[Node]:
        # The path generated from start_tree ends at q_init, but we want it to start at q_init. So we must reverse it.
        # The path generated from goal_tree ends at q_goal, which is what we want.
        path_start = start_tree.get_path()
        path_start.reverse()
        path_end = goal_tree.get_path()
        return path_start + path_end

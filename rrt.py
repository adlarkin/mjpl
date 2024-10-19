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
    lower_limits = [ model.joint(name).range[0] for name in joint_names ]
    upper_limits = [ model.joint(name).range[1] for name in joint_names ]
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
        # TODO: use something like a kd-tree to improve nearest neighbor lookup times
        # (this will impact how nearest_neighbor is implemented)
        self.nodes = set()

    def add_node(self, node):
        self.nodes.add(node)

    def nearest_neighbor(self, q):
        closest_node = None
        min_dist = np.inf
        for n in self.nodes:
            neighboring_dist = configuration_distance(n.q, q)
            if neighboring_dist < min_dist:
                closest_node = n
                min_dist = neighboring_dist
        return closest_node

    def get_path(self, end_node: Node):
        path = []
        curr_node = end_node
        while curr_node.parent is not None:
            path.append(curr_node.q)
            curr_node = curr_node.parent
        path.reverse()
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
        tree = Tree()
        tree.add_node(Node(q_init, None))

        if not is_valid_config(q_goal, self.joint_limits_lower, self.joint_limits_upper, self.joint_qpos_addrs, self.model, self.data):
            print("q_goal is not a valid configuration")
            return []

        solution_found = False

        # Is there a direct connection to q_goal from q_init?
        if configuration_distance(q_init, q_goal) <= self.options.epsilon:
            tree.add_node(Node(q_goal, q_init))
            solution_found = True

        max_planning_time = float('inf') if self.options.max_planning_time <= 0 else self.options.max_planning_time
        start_time = time.time()
        while (not solution_found and time.time() - start_time < max_planning_time):
            if self.options.rng.random() <= self.options.goal_biasing_probability:
                q_rand = q_goal
            else:
                q_rand = random_config(self.options.rng, self.joint_limits_lower, self.joint_limits_upper)
            next_node = self.extend(q_rand, self.options.epsilon, tree)
            if is_valid_config(next_node.q, self.joint_limits_lower, self.joint_limits_upper, self.joint_qpos_addrs, self.model, self.data):
                tree.add_node(next_node)
                # check if the latest node yields a connection to q_goal
                if configuration_distance(next_node.q, q_goal) <= self.options.epsilon:
                    tree.add_node(Node(q_goal, next_node))
                    solution_found = True
        print(f"Solution found: {solution_found}, time taken: {time.time() - start_time}")
        if solution_found:
            return tree.get_path(tree.nearest_neighbor(q_goal))
        return []

    def extend(self, q, epsilon: float, tree: Tree) -> Node:
        node_near = tree.nearest_neighbor(q)
        q_near = node_near.q
        q_dist = configuration_distance(q_near, q)
        if q_dist <= epsilon:
            return Node(q, node_near)
        q_increment = epsilon * ((q - q_near) / q_dist)
        return Node(q_near + q_increment, node_near)

import copy
import mujoco
import numpy as np
import time


def configuration_distance(q_from, q_to):
    return np.linalg.norm(q_to - q_from)


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


class RRT:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # The model should be read-only, so we don't need to make a deep copy of it.
        self.model = model

        # We need the current state of the scene for collision checking, but we
        # also need to check joint configurations via FK to ensure that newly
        # sampled states are valid. 
        self.data = copy.deepcopy(data)

        # TODO: don't hardcode this (currently assumes franka panda)
        self.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
        ]
        self.joint_qpos_addrs = np.array([ model.joint(j_name).qposadr.item() for j_name in self.joint_names ])
        self.joint_limits_lower = [ model.joint(j_name).range[0] for j_name in self.joint_names ]
        self.joint_limits_upper = [ model.joint(j_name).range[1] for j_name in self.joint_names ]

        # Random number generator that's used for sampling joint configurations.
        self.rng = np.random.default_rng()

    def build_tree(self, q_goal):
        print("Starting to plan...")

        # Assuming that MjData's current joint configuration is valid, set it as the root of the tree.
        q_init = [ self.data.joint(j_name).qpos.item() for j_name in self.joint_names ]
        if not self.is_valid_config(q_init):
            print("q_init is not a valid configuration")
        tree = Tree()
        tree.add_node(Node(q_init, None))

        if not self.is_valid_config(q_goal):
            print("q_goal is not a valid configuration")

        # TODO: remove hardcoded params
        max_planning_time = 10
        epsilon = 0.05
        goal_biasing_probability = 0.1

        solution_found = False

        # Is there a direct connection to q_goal from q_init?
        if configuration_distance(q_init, q_goal) <= epsilon:
            tree.add_node(Node(q_goal, q_init))
            solution_found = True

        start_time = time.time()
        while (not solution_found and time.time() - start_time < max_planning_time):
            if self.rng.random() <= goal_biasing_probability:
                q_rand = q_goal
            else:
                q_rand = self.random_config()
            next_node = self.extend(q_rand, epsilon, tree)
            if self.is_valid_config(next_node.q):
                tree.add_node(next_node)
                # check if the latest node yields a connection to q_goal
                if configuration_distance(next_node.q, q_goal) <= epsilon:
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

    def random_config(self):
        return self.rng.uniform(low=self.joint_limits_lower, high=self.joint_limits_upper)

    def is_valid_config(self, q) -> bool:
        # We assume that a valid joint config is one that is collision free.
        # Since random_config() already samples within the joint limits and extend() "clips"
        # sampled configurations, we shouldn't have to check if new joint configurations
        # violate joint limits or not.
        #
        # In order to check if q is a valid joint config for the given scene, we need to run
        # FK on the new values before doing collision checking.
        '''
        # TODO: find a way to map joint names to data.qpos so that this isn't hardcoded
        self.data.qpos[:7] = q
        '''
        '''
        # TODO: improve the runtime performance here so that a for loop isn't needed.
        # Probably need to pre-compute the idxs in data.qpos
        for idx in range(len(self.joint_names)):
            self.data.joint(self.joint_names[idx]).qpos = q[idx]
        '''
        self.data.qpos[self.joint_qpos_addrs] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        return not self.data.ncon

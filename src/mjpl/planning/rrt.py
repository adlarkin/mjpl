import time

import mujoco
import numpy as np
from mink.lie.se3 import SE3

from .. import utils
from ..collision_ruleset import CollisionRuleset
from ..inverse_kinematics.ik_solver import IKSolver
from ..inverse_kinematics.mink_ik_solver import MinkIKSolver
from ..joint_group import JointGroup
from .tree import Node, Tree
from .utils import _combine_paths, _connect


class RRT:
    """Bi-directional RRT-Connect.

    This implementation runs CONNECT on both trees. The original algorithm runs CONNECT
    on one tree and EXTEND on the other, swapping trees every iteration:
    https://www.cs.cmu.edu/afs/cs/academic/class/15494-s14/readings/kuffner_icra2000.pdf
    """

    def __init__(
        self,
        jg: JointGroup,
        cr: CollisionRuleset,
        max_planning_time: float = 10.0,
        epsilon: float = 0.05,
        seed: int | None = None,
        goal_biasing_probability: float = 0.05,
        max_connection_distance: float = np.inf,
    ) -> None:
        """Constructor.

        Args:
            jg: The JointGroup used for planning.
            cr: The CollisionRuleset the sampled configurations must obey.
            max_planning_time: Maximum planning time, in seconds.
            epsilon: The maximum distance allowed between nodes in the tree.
            seed: Seed used for the underlying sampler in the planner.
                `None` means the algorithm is nondeterministc.
            goal_biasing_probability: Probability of sampling a goal state during planning.
                This must be a value between [0.0, 1.0].
            max_connection_distance: The maximum distance for extending a tree using CONNECT.
        """
        if (
            max_planning_time < 0.0
            or epsilon <= 0.0
            or max_connection_distance < 0.0
            or goal_biasing_probability < 0.0
            or goal_biasing_probability > 1.0
        ):
            raise ValueError(
                "`max_planning_time` and `max_connection_distance` must be >= 0.0. "
                "`epsilon` must be > 0.0. `goal_biasing_probability` must be within [0.0, 1.0]."
            )
        self.jg = jg
        self.cr = cr
        self.max_planning_time = max_planning_time
        self.epsilon = epsilon
        self.seed = seed
        self.goal_biasing_probability = goal_biasing_probability
        self.max_connection_distance = max_connection_distance

        self.rng = np.random.default_rng(seed=seed)

    def plan_to_pose(
        self,
        q_init_world: np.ndarray,
        pose: SE3,
        site: str,
        solver: IKSolver | None = None,
    ) -> list[np.ndarray]:
        """Plan to a pose.

        Args:
            q_init_world: Initial joint configuration of the world.
            pose: Target pose, in the world frame.
            site: The site (i.e., frame) that must satisfy `pose`.
            solver: Solver used to compute IK for `pose` and `site`.

        Returns:
            A path from `q_init_world` to `pose`. If a path cannot be found, an
            empty list is returned.

            A path is defined as a list of configurations that correspond to the
            joints in the planner's JointGroup.
        """
        return self.plan_to_poses(q_init_world, [pose], site, solver)

    def plan_to_config(
        self, q_init_world: np.ndarray, q_goal: np.ndarray
    ) -> list[np.ndarray]:
        """Plan to a configuration.

        Args:
            q_init_world: Initial joint configuration of the world.
            q_goals: Goal joint configuration, which should specify values for
                each joint in the planner's JointGroup.

        Returns:
            A path from `q_init_world` to `q_goal`. If a path cannot be found, an
            empty list is returned.

            A path is defined as a list of configurations that correspond to the
            joints in the planner's joint group.
        """
        return self.plan_to_configs(q_init_world, [q_goal])

    def plan_to_poses(
        self,
        q_init_world: np.ndarray,
        poses: list[SE3],
        site: str,
        solver: IKSolver | None = None,
    ) -> list[np.ndarray]:
        """Plan to a list of poses.

        Args:
            q_init_world: Initial joint configuration of the world.
            poses: Target poses, in the world frame.
            site: The site (i.e., frame) that must satisfy each pose in `poses`.
            solver: Solver used to compute IK for `poses` and `site`.

        Returns:
            A path from `q_init_world` to a pose in `poses`. The planner will
            return the first path that is found. If a path cannot be found to
            any of the poses, an empty list is returned.

            A path is defined as a list of configurations that correspond to the
            joints in the planner's JointGroup.
        """
        if solver is None:
            solver = MinkIKSolver(
                model=self.jg.model,
                jg=self.jg,
                cr=self.cr,
                seed=self.seed,
                max_attempts=5,
            )
        potential_solutions = [
            solver.solve_ik(p, site, q_init_guess=q_init_world) for p in poses
        ]
        valid_solutions = [q for q in potential_solutions if q is not None]
        if not valid_solutions:
            print("Unable to find at least one configuration from the target poses.")
            return []

        goal_configs = [q[self.jg.qpos_addrs] for q in valid_solutions]
        return self.plan_to_configs(q_init_world, goal_configs)

    def plan_to_configs(
        self, q_init_world: np.ndarray, q_goals: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Plan to a list of configurations.

        Args:
            q_init_world: Initial joint configuration of the world.
            q_goals: Goal joint configurations, which should specify values for
                each joint in the planner's JointGroup.

        Returns:
            A path from `q_init_world` to a configuration in `q_goals`. The
            planner will return the first path that is found. If a path cannot
            be found to any of the configurations, an empty list is returned.

            A path is defined as a list of configurations that correspond to the
            joints in the planner's JointGroup.
        """
        assert q_init_world.size == self.jg.model.nq
        for q in q_goals:
            assert q.size == len(self.jg.joint_ids)

        data = mujoco.MjData(self.jg.model)
        data.qpos = q_init_world
        q_init = self.jg.qpos(data)
        if not utils.is_valid_config(q_init, self.jg, data, self.cr):
            print("q_init is not a valid configuration")
            return []
        for q in q_goals:
            if not utils.is_valid_config(q, self.jg, data, self.cr):
                print(f"The following goal config is not a valid configuration: {q}")
                return []

        # Is there a direct connection to any of the goals from q_init?
        for q in q_goals:
            if np.linalg.norm(q - q_init) <= self.epsilon:
                return [q_init, q]

        start_tree = Tree(Node(q_init))
        # To support multiple goals, the root of the goal tree is a sink node
        # (i.e., a node with an empty numpy array) and all goal configs are
        # children of this sink node.
        sink_node = Node(np.array([]))
        goal_nodes = [Node(q, sink_node) for q in q_goals]
        goal_tree = Tree(sink_node, is_sink=True)
        for n in goal_nodes:
            goal_tree.add_node(n)

        start_time = time.time()
        while time.time() - start_time < self.max_planning_time:
            if self.rng.random() <= self.goal_biasing_probability:
                # randomly pick a goal
                random_goal_idx = self.rng.integers(0, len(goal_nodes))
                q_rand = goal_nodes[random_goal_idx].q
            else:
                q_rand = self.jg.random_config(self.rng)

            new_start_tree_node = _connect(
                q_rand,
                start_tree,
                self.epsilon,
                self.max_connection_distance,
                self.jg,
                self.cr,
                data,
            )
            new_goal_tree_node = _connect(
                new_start_tree_node.q,
                goal_tree,
                self.epsilon,
                self.max_connection_distance,
                self.jg,
                self.cr,
                data,
            )
            if (
                np.linalg.norm(new_goal_tree_node.q - new_start_tree_node.q)
                <= self.epsilon
            ):
                return _combine_paths(
                    start_tree, new_start_tree_node, goal_tree, new_goal_tree_node
                )

            if not np.array_equal(new_start_tree_node.q, q_rand):
                new_goal_tree_node = _connect(
                    q_rand,
                    goal_tree,
                    self.epsilon,
                    self.max_connection_distance,
                    self.jg,
                    self.cr,
                    data,
                )
                new_start_tree_node = _connect(
                    new_goal_tree_node.q,
                    start_tree,
                    self.epsilon,
                    self.max_connection_distance,
                    self.jg,
                    self.cr,
                    data,
                )
                if (
                    np.linalg.norm(new_goal_tree_node.q - new_start_tree_node.q)
                    <= self.epsilon
                ):
                    return _combine_paths(
                        start_tree, new_start_tree_node, goal_tree, new_goal_tree_node
                    )

        return []

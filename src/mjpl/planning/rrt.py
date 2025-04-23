import time

import mujoco
import numpy as np
from mink.lie.se3 import SE3

from .. import utils
from ..collision_ruleset import CollisionRuleset
from ..inverse_kinematics.ik_solver import IKSolver
from ..inverse_kinematics.mink_ik_solver import MinkIKSolver
from ..types import Path
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
        model: mujoco.MjModel,
        planning_joints: list[str],
        cr: CollisionRuleset,
        max_planning_time: float = 10.0,
        epsilon: float = 0.05,
        seed: int | None = None,
        goal_biasing_probability: float = 0.05,
        max_connection_distance: float = np.inf,
    ) -> None:
        """Constructor.

        Args:
            model: MuJoCo model.
            planning_joints: The joints used for planning. An empty list means all
                joints will be used.
            cr: The CollisionRuleset the sampled configurations must obey.
            max_planning_time: Maximum planning time, in seconds.
            epsilon: The maximum distance allowed between nodes in the tree.
            seed: Seed used for the underlying sampler in the planner.
                `None` means the algorithm is nondeterministc.
            goal_biasing_probability: Probability of sampling a goal state during planning.
                This must be a value between [0.0, 1.0].
            max_connection_distance: The maximum distance for extending a tree using CONNECT.
        """
        if max_planning_time <= 0.0:
            raise ValueError("`max_planning_time` must be > 0.0")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.0")
        if max_connection_distance <= 0.0:
            raise ValueError("`max_connection_distance` must be > 0.0")
        if goal_biasing_probability < 0.0 or goal_biasing_probability > 1.0:
            raise ValueError("`goal_biasing_probability` must be within [0.0, 1.0].")

        self.model = model
        self.planning_joints = planning_joints
        self.q_idx = utils.qpos_idx(model, planning_joints, default_to_full=True)
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
    ) -> Path | None:
        """Plan to a pose.

        Args:
            q_init_world: Initial joint configuration of the world.
            pose: Target pose, in the world frame.
            site: The site (i.e., frame) that must satisfy `pose`.
            solver: Solver used to compute IK for `pose` and `site`.

        Returns:
            A path from `q_init_world` to `pose`. If a path cannot be found,
            None is returned.
        """
        return self.plan_to_poses(q_init_world, [pose], site, solver)

    def plan_to_config(
        self, q_init_world: np.ndarray, q_goal: np.ndarray
    ) -> Path | None:
        """Plan to a configuration.

        Args:
            q_init_world: Initial joint configuration of the world.
            q_goals: Goal joint configuration, which should specify values for
                each joint in the planner's JointGroup.

        Returns:
            A path from `q_init_world` to `pose`. If a path cannot be found,
            None is returned.
        """
        return self.plan_to_configs(q_init_world, [q_goal])

    def plan_to_poses(
        self,
        q_init_world: np.ndarray,
        poses: list[SE3],
        site: str,
        solver: IKSolver | None = None,
    ) -> Path | None:
        """Plan to a list of poses.

        Args:
            q_init_world: Initial joint configuration of the world.
            poses: Target poses, in the world frame.
            site: The site (i.e., frame) that must satisfy each pose in `poses`.
            solver: Solver used to compute IK for `poses` and `site`.

        Returns:
            A path from `q_init_world` to `pose`. If a path cannot be found,
            None is returned.
        """
        if solver is None:
            solver = MinkIKSolver(
                model=self.model,
                joints=self.planning_joints,
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
            return None

        goal_configs = [q[self.q_idx] for q in valid_solutions]
        return self.plan_to_configs(q_init_world, goal_configs)

    def plan_to_configs(
        self, q_init_world: np.ndarray, q_goals: list[np.ndarray]
    ) -> Path | None:
        """Plan to a list of configurations.

        Args:
            q_init_world: Initial joint configuration of the world.
            q_goals: Goal joint configurations, which should specify values for
                each joint in the planner's JointGroup.

        Returns:
            A path from `q_init_world` to a configuration in `q_goals`. The
            planner will return the first path that is found. If a path cannot
            be found to any of the configurations, None is returned.
        """
        assert q_init_world.size == self.model.nq
        for q in q_goals:
            assert q.size == len(self.q_idx)

        data = mujoco.MjData(self.model)
        data.qpos = q_init_world
        if not utils.is_valid_config(self.model, data, self.cr):
            print("q_init_world is not a valid configuration")
            return None
        for q in q_goals:
            data.qpos[self.q_idx] = q
            if not utils.is_valid_config(self.model, data, self.cr):
                print(f"The following goal config is not a valid configuration: {q}")
                return None

        # Is there a direct connection to any of the goals from q_init?
        q_init = q_init_world[self.q_idx]
        for q in q_goals:
            if np.linalg.norm(q - q_init) <= self.epsilon:
                return Path(
                    q_init=q_init_world,
                    waypoints=[q_init, q],
                    joints=self.planning_joints,
                )

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
                q_rand = self.rng.uniform(*self.model.jnt_range.T)[self.q_idx]

            new_start_tree_node = _connect(
                self.model,
                data,
                q_rand,
                self.q_idx,
                start_tree,
                self.epsilon,
                self.max_connection_distance,
                self.cr,
            )
            new_goal_tree_node = _connect(
                self.model,
                data,
                new_start_tree_node.q,
                self.q_idx,
                goal_tree,
                self.epsilon,
                self.max_connection_distance,
                self.cr,
            )
            if (
                np.linalg.norm(new_goal_tree_node.q - new_start_tree_node.q)
                <= self.epsilon
            ):
                waypoints = _combine_paths(
                    start_tree, new_start_tree_node, goal_tree, new_goal_tree_node
                )
                return Path(
                    q_init=q_init_world,
                    waypoints=waypoints,
                    joints=self.planning_joints,
                )

            if not np.array_equal(new_start_tree_node.q, q_rand):
                new_goal_tree_node = _connect(
                    self.model,
                    data,
                    q_rand,
                    self.q_idx,
                    goal_tree,
                    self.epsilon,
                    self.max_connection_distance,
                    self.cr,
                )
                new_start_tree_node = _connect(
                    self.model,
                    data,
                    new_goal_tree_node.q,
                    self.q_idx,
                    start_tree,
                    self.epsilon,
                    self.max_connection_distance,
                    self.cr,
                )
                if (
                    np.linalg.norm(new_goal_tree_node.q - new_start_tree_node.q)
                    <= self.epsilon
                ):
                    waypoints = _combine_paths(
                        start_tree, new_start_tree_node, goal_tree, new_goal_tree_node
                    )
                    return Path(
                        q_init=q_init_world,
                        waypoints=waypoints,
                        joints=self.planning_joints,
                    )

        return None

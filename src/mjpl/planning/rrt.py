import time

import mujoco
import numpy as np
from mink.lie.se3 import SE3

from .. import utils
from ..constraint.constraint_interface import Constraint
from ..constraint.utils import obeys_constraints
from ..inverse_kinematics.ik_solver import IKSolver
from ..inverse_kinematics.mink_ik_solver import MinkIKSolver
from ..types import Path
from .tree import Node, Tree
from .utils import _combine_paths, _connect


class RRT:
    """Bi-directional RRT, with support for constraints.

    This implementation runs CONNECT on both trees. The original algorithm runs CONNECT
    on one tree and EXTEND on the other, swapping trees every iteration.

    References:
    - https://www.cs.cmu.edu/afs/cs/academic/class/15494-s14/readings/kuffner_icra2000.pdf
    - https://personalrobotics.cs.washington.edu/publications/berenson2009cbirrt.pdf
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        planning_joints: list[str],
        constraints: list[Constraint],
        max_planning_time: float = 10.0,
        epsilon: float = 0.05,
        seed: int | None = None,
        goal_biasing_probability: float = 0.05,
        max_connection_distance: float = np.inf,
    ) -> None:
        """Constructor.

        Args:
            model: MuJoCo model.
            planning_joints: The joints that are sampled during planning.
            constraints: The constraints the sampled configurations must obey.
            max_planning_time: Maximum planning time, in seconds.
            epsilon: The maximum distance allowed between nodes in the tree.
            seed: Seed used for the underlying sampler in the planner.
                `None` means the algorithm is nondeterministc.
            goal_biasing_probability: Probability of sampling a goal state during planning.
                This must be a value between [0.0, 1.0].
            max_connection_distance: The maximum distance for extending a tree using CONNECT.
        """
        if not planning_joints:
            raise ValueError("`planning_joints` cannot be empty.")
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
        self.constraints = constraints
        self.max_planning_time = max_planning_time
        self.epsilon = epsilon
        self.seed = seed
        self.goal_biasing_probability = goal_biasing_probability
        self.max_connection_distance = max_connection_distance

    def plan_to_pose(
        self,
        q_init: np.ndarray,
        pose: SE3,
        site: str,
        solver: IKSolver | None = None,
    ) -> Path | None:
        """Plan to a pose.

        Args:
            q_init: Initial joint configuration.
            pose: Target pose, in the world frame.
            site: The site (i.e., frame) that must satisfy `pose`.
            solver: Solver used to compute IK for `pose` and `site`.

        Returns:
            A path from `q_init` to `pose`. If a path cannot be found,
            None is returned.
        """
        return self.plan_to_poses(q_init, [pose], site, solver)

    def plan_to_config(self, q_init: np.ndarray, q_goal: np.ndarray) -> Path | None:
        """Plan to a configuration.

        Args:
            q_init: Initial joint configuration.
            q_goals: Goal joint configuration.

        Returns:
            A path from `q_init` to `pose`. If a path cannot be found,
            None is returned.
        """
        return self.plan_to_configs(q_init, [q_goal])

    def plan_to_poses(
        self,
        q_init: np.ndarray,
        poses: list[SE3],
        site: str,
        solver: IKSolver | None = None,
    ) -> Path | None:
        """Plan to a list of poses.

        Args:
            q_init: Initial joint configuration.
            poses: Target poses, in the world frame.
            site: The site (i.e., frame) that must satisfy each pose in `poses`.
            solver: Solver used to compute IK for `poses` and `site`.

        Returns:
            A path from `q_init` to `pose`. If a path cannot be found,
            None is returned.
        """
        if solver is None:
            solver = MinkIKSolver(
                model=self.model,
                joints=self.planning_joints,
                constraints=self.constraints,
                seed=self.seed,
                max_attempts=5,
            )
        potential_solutions = [
            solver.solve_ik(p, site, q_init_guess=q_init) for p in poses
        ]
        valid_solutions = [q for q in potential_solutions if q is not None]
        if not valid_solutions:
            print("Unable to find at least one configuration from the target poses.")
            return None
        return self.plan_to_configs(q_init, valid_solutions)

    def plan_to_configs(
        self, q_init: np.ndarray, q_goals: list[np.ndarray]
    ) -> Path | None:
        """Plan to a list of configurations.

        Args:
            q_init: Initial joint configuration.
            q_goals: Goal joint configurations.

        Returns:
            A path from `q_init` to a configuration in `q_goals`. The
            planner will return the first path that is found. If a path cannot
            be found to any of the configurations, None is returned.
        """
        if not obeys_constraints(q_init, self.constraints):
            raise ValueError("q_init is not a valid configuration")
        for q in q_goals:
            if not obeys_constraints(q, self.constraints):
                raise ValueError(
                    f"The following goal config is not a valid configuration: {q}"
                )

        q_idx = utils.qpos_idx(self.model, self.planning_joints)
        fixed_jnt_idx = [i for i in range(self.model.nq) if i not in q_idx]
        for q in q_goals:
            if not np.allclose(
                q_init[fixed_jnt_idx], q[fixed_jnt_idx], rtol=0, atol=1e-12
            ):
                raise ValueError(
                    f"The following goal config has values for joints outside of "
                    f"the planner's planning joints that don't match q_init: {q}. "
                    f"q_init is {q_init}, and the planning joints are {self.planning_joints}"
                )

        # Is there a direct connection to any of the goals from q_init?
        for q in q_goals:
            if np.linalg.norm(q - q_init) <= self.epsilon:
                return Path(
                    q_init=q_init,
                    waypoints=[q_init, q],
                    joints=utils.all_joints(self.model),
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

        rng = np.random.default_rng(seed=self.seed)
        start_time = time.time()
        while time.time() - start_time < self.max_planning_time:
            if rng.random() <= self.goal_biasing_probability:
                # Randomly pick a goal.
                random_goal_idx = rng.integers(0, len(goal_nodes))
                q_rand = goal_nodes[random_goal_idx].q
            else:
                # Create a random configuration.
                q_rand = q_init.copy()
                q_rand[q_idx] = rng.uniform(*self.model.jnt_range.T)[q_idx]

            new_start_tree_node = _connect(
                q_rand,
                start_tree,
                self.epsilon,
                self.max_connection_distance,
                self.constraints,
            )
            new_goal_tree_node = _connect(
                new_start_tree_node.q,
                goal_tree,
                self.epsilon,
                self.max_connection_distance,
                self.constraints,
            )
            if new_start_tree_node == new_goal_tree_node:
                waypoints = _combine_paths(
                    start_tree, new_start_tree_node, goal_tree, new_goal_tree_node
                )
                return Path(
                    q_init=q_init,
                    waypoints=waypoints,
                    joints=utils.all_joints(self.model),
                )

            # If the start tree was not able to reach q_rand, try the opposite process
            # (grow the goal tree towards q_rand first). This can help reduce bias in
            # each tree's growth.
            if not np.array_equal(new_start_tree_node.q, q_rand):
                new_goal_tree_node = _connect(
                    q_rand,
                    goal_tree,
                    self.epsilon,
                    self.max_connection_distance,
                    self.constraints,
                )
                new_start_tree_node = _connect(
                    new_goal_tree_node.q,
                    start_tree,
                    self.epsilon,
                    self.max_connection_distance,
                    self.constraints,
                )
                if new_start_tree_node == new_goal_tree_node:
                    waypoints = _combine_paths(
                        start_tree, new_start_tree_node, goal_tree, new_goal_tree_node
                    )
                    return Path(
                        q_init=q_init,
                        waypoints=waypoints,
                        joints=utils.all_joints(self.model),
                    )

        return None

import time

import mujoco
import numpy as np
from mink.lie.se3 import SE3

from .. import utils
from ..constraint.constraint_interface import Constraint
from ..constraint.utils import obeys_constraints
from ..inverse_kinematics.ik_solver_interface import IKSolver
from ..inverse_kinematics.mink_ik_solver import MinkIKSolver
from .tree import Node, Tree
from .utils import _combine_paths, _constrained_extend


class RRT:
    """CBiRRT: Bi-directional RRT, with support for constraints.

    Reference: https://personalrobotics.cs.washington.edu/publications/berenson2009cbirrt.pdf
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
        """
        if not planning_joints:
            raise ValueError("`planning_joints` cannot be empty.")
        if max_planning_time <= 0.0:
            raise ValueError("`max_planning_time` must be > 0.0")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.0")
        if goal_biasing_probability < 0.0 or goal_biasing_probability > 1.0:
            raise ValueError("`goal_biasing_probability` must be within [0.0, 1.0].")

        self.model = model
        self.planning_joints = planning_joints
        self.constraints = constraints
        self.max_planning_time = max_planning_time
        self.epsilon = epsilon
        self.seed = seed
        self.goal_biasing_probability = goal_biasing_probability

    def plan_to_pose(
        self,
        q_init: np.ndarray,
        pose: SE3,
        site: str,
        solver: IKSolver | None = None,
    ) -> list[np.ndarray]:
        """Plan to a pose.

        Args:
            q_init: Initial joint configuration.
            pose: Target pose, in the world frame.
            site: The site (i.e., frame) that must satisfy `pose`.
            solver: Solver used to compute IK for `pose` and `site`.

        Returns:
            A list of waypoints that form a path from `q_init` to a configuration
            that satisfies the specified pose. If a path cannot be found, an empty
            list is returned.
        """
        return self.plan_to_poses(q_init, [pose], site, solver)

    def plan_to_config(
        self, q_init: np.ndarray, q_goal: np.ndarray
    ) -> list[np.ndarray]:
        """Plan to a configuration.

        Args:
            q_init: Initial joint configuration.
            q_goal: Goal joint configuration.

        Returns:
            A list of waypoints that form a path from `q_init` to `q_goal`.
            If a path cannot be found, an empty list is returned.
        """
        return self.plan_to_configs(q_init, [q_goal])

    def plan_to_poses(
        self,
        q_init: np.ndarray,
        poses: list[SE3],
        site: str,
        solver: IKSolver | None = None,
    ) -> list[np.ndarray]:
        """Plan to a list of poses.

        Args:
            q_init: Initial joint configuration.
            poses: Target poses, in the world frame.
            site: The site (i.e., frame) that must satisfy each pose in `poses`.
            solver: Solver used to compute IK for `poses` and `site`.

        Returns:
            A list of waypoints that form a path from `q_init` to a configuration
            that satisfies a pose in `poses`. If a path cannot be found, an empty
            list is returned.
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
            return []
        return self.plan_to_configs(q_init, valid_solutions)

    def plan_to_configs(
        self, q_init: np.ndarray, q_goals: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Plan to a list of configurations.

        Args:
            q_init: Initial joint configuration.
            q_goals: Goal joint configurations.

        Returns:
            A list of waypoints that form a path from `q_init` to a goal in `q_goals`.
            If a path cannot be found, an empty list is returned.
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

        rng = np.random.default_rng(seed=self.seed)
        tree_a, tree_b = start_tree, goal_tree
        swapped = False

        start_time = time.time()
        while time.time() - start_time < self.max_planning_time:
            if rng.random() <= self.goal_biasing_probability:
                if swapped:
                    q_rand = q_init
                else:
                    # Randomly pick a goal.
                    random_goal_idx = rng.integers(0, len(goal_nodes))
                    q_rand = goal_nodes[random_goal_idx].q
            else:
                # Create a random configuration.
                q_rand = q_init.copy()
                q_rand[q_idx] = rng.uniform(*self.model.jnt_range.T)[q_idx]

            # Run constrained extend on both trees.
            q_reached_a = _constrained_extend(
                q_rand,
                tree_a,
                self.epsilon,
                self.constraints,
            )
            q_reached_b = _constrained_extend(
                q_reached_a,
                tree_b,
                self.epsilon,
                self.constraints,
            )
            if np.array_equal(q_reached_a, q_reached_b):
                return _combine_paths(
                    start_tree,
                    start_tree.nearest_neighbor(q_reached_a),
                    goal_tree,
                    goal_tree.nearest_neighbor(q_reached_a),
                )

            # Swap trees.
            tree_a, tree_b = tree_b, tree_a
            swapped = not swapped

        return []

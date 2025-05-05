import mink
import mujoco
import numpy as np

from .constraint.constraint_interface import Constraint
from .constraint.utils import apply_constraints, obeys_constraints


def step(start: np.ndarray, target: np.ndarray, max_step_dist: float) -> np.ndarray:
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
        return target.copy()
    direction = target - start
    magnitude = np.linalg.norm(direction)
    unit_vec = direction / magnitude
    return start + (unit_vec * min(max_step_dist, magnitude))


def all_joints(model: mujoco.MjModel) -> list[str]:
    """Get all joint names in a MuJoCo model.

    Args:
        model: MuJoCo model.

    Returns:
        A list of all of the joint names in `model`.
    """
    return [model.joint(j).name for j in range(model.njnt)]


def qpos_idx(model: mujoco.MjModel, joints: list[str]) -> list[int]:
    """Get the indices in mujoco.MjData.qpos that correspond to specific joints.

    Args:
        model: MuJoCo model.
        joints: The names of the joints in `model`.

    Returns:
        A list of indices that correspond to `joints` in mujoco.MjData.qpos.
    """
    idx: list[int] = []
    for j in joints:
        jnt_id = model.joint(j).id
        jnt_dim = mink.constants.qpos_width(model.jnt_type[jnt_id])
        q_start = model.jnt_qposadr[jnt_id]
        idx.extend(range(q_start, q_start + jnt_dim))
    return idx


def qvel_idx(model: mujoco.MjModel, joints: list[str]) -> list[int]:
    """Get the indices in mujoco.MjData.qvel that correspond to specific joints.

    Args:
        model: MuJoCo model.
        joints: The names of the joints in `model`.

    Returns:
        A list of indices that correspond to `joints` in mujoco.MjData.qvel.
    """
    idx: list[int] = []
    for j in joints:
        jnt_id = model.joint(j).id
        jnt_dim = mink.constants.dof_width(model.jnt_type[jnt_id])
        vel_start = model.jnt_dofadr[jnt_id]
        idx.extend(range(vel_start, vel_start + jnt_dim))
    return idx


def site_pose(data: mujoco.MjData, site_name: str) -> mink.SE3:
    """Get the pose of a site in the world frame.

    Args:
        data: MuJoCo data.
        site_name: The name of the site.

    Returns:
        The pose of the site in the world frame.
    """
    position = data.site(site_name).xpos.copy()
    rotation = data.site(site_name).xmat.copy()
    return mink.SE3.from_rotation_and_translation(
        mink.SO3.from_matrix(rotation.reshape(3, 3)),
        position,
    )


def random_config(
    model: mujoco.MjModel,
    q_init: np.ndarray,
    joints: list[str],
    seed: int | None = None,
    constraints: list[Constraint] = [],
) -> np.ndarray:
    """Generate a random configuration that obeys constraints.

    Args:
        model: MuJoCo model.
        q_init: Initial joint configuration. Used to set values for joints that are
            not in `joints`.
        joints: The joints to set random values for.
        seed: Seed used for random number generation.
        constraints: Constraints the randomly generated configuration must obey.
            Set this to an empty list if no constraints should be enforced.

    Returns:
        A random configuration that obeys `constraints`.
    """
    q_idx = qpos_idx(model, joints)
    rng = np.random.default_rng(seed=seed)

    q = q_init.copy()
    q[q_idx] = rng.uniform(*model.jnt_range.T)[q_idx]

    q_constrained = apply_constraints(q, constraints)
    while q_constrained is None:
        q[q_idx] = rng.uniform(*model.jnt_range.T)[q_idx]
        q_constrained = apply_constraints(q, constraints)
    return q_constrained


def shortcut(
    waypoints: list[np.ndarray],
    constraints: list[Constraint],
    validation_dist: float = 0.05,
    max_attempts: int = 100,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Perform shortcutting on a list of waypoints.

    Args:
        waypoints: The waypoints to shortcut.
        constraints: The constraints to enforce (if any) for validation checks.
        validation_dist: The distance between each validation check, which occurs
            between a pair of waypoints that are trying to be directly connected
            if these waypoints are further than `validation_dist` apart.
        max_attempts: The maximum number of shortcut attempts. Each attempt will
            randomly select two waypoints to connect. If exactly two waypoints
            remain, no more attempts will be executed.
        seed: The seed which is used for randomly picking pairs of waypoints
            to shortcut.

    Returns:
        A waypoint list with direct connections between each adjacent waypoint that
        obeys `constraints`.
    """
    rng = np.random.default_rng(seed=seed)

    # sanity check: can we shortcut directly between the start/end of the path?
    shortened_waypoints = _connect_waypoints(
        waypoints,
        start_idx=0,
        end_idx=len(waypoints) - 1,
        validation_dist=validation_dist,
        constraints=constraints,
    )
    for _ in range(max_attempts):
        if len(shortened_waypoints) == 2:
            # we can go directly from start to goal, so no more shortcutting can be done
            return shortened_waypoints
        # randomly pick 2 waypoints
        start, end = 0, 0
        while start == end:
            start, end = rng.integers(len(shortened_waypoints), size=2)
        if start > end:
            start, end = end, start
        shortened_waypoints = _connect_waypoints(
            shortened_waypoints,
            start_idx=start,
            end_idx=end,
            validation_dist=validation_dist,
            constraints=constraints,
        )

    return shortened_waypoints


def _connect_waypoints(
    waypoints: list[np.ndarray],
    start_idx: int,
    end_idx: int,
    validation_dist: float,
    constraints: list[Constraint] = [],
) -> list[np.ndarray]:
    """If possible, directly connect two specific waypoints from a list of waypoints.

    Args:
        waypoints: The list of waypoints.
        start_idx: The index of the first waypoint.
        end_idx: The index of the second waypoint.
        validation_dist: The distance increment used for performing intermediate
            validation checks. This must be > 0.
        constraints: The constraints to enforce (if any) for validation checks.

    Returns:
        A waypoint list with a direct connection between the waypoints at indices
        (`start_idx`, `end_idx`) if the waypoints at these indices can be connected
        without violating `constraints`.
    """
    if validation_dist <= 0.0:
        raise ValueError("`validation_dist` must be > 0")

    q_start = waypoints[start_idx]
    q_target = waypoints[end_idx]

    q_curr = step(q_start, q_target, validation_dist)
    while not np.array_equal(q_curr, q_target):
        if not obeys_constraints(q_curr, constraints):
            return waypoints
        q_curr = step(q_curr, q_target, validation_dist)
    return waypoints[: start_idx + 1] + waypoints[end_idx:]

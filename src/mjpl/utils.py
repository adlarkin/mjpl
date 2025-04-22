import mink
import mujoco
import numpy as np

from .collision_ruleset import CollisionRuleset
from .types import Path


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


def is_valid_config(
    q: np.ndarray,
    model: mujoco.MjModel,
    q_idx: list[int] = [],
    cr: CollisionRuleset | None = None,
    data: mujoco.MjData | None = None,
) -> bool:
    q_idx = q_idx or list(range(model.nq))

    # Check joint limits.
    if not np.all((q >= model.jnt_range[q_idx, 0]) & (q <= model.jnt_range[q_idx, 1])):
        return False

    if cr:
        # Check for collisions.
        data = data or mujoco.MjData(model)
        data.qpos[q_idx] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        return cr.obeys_ruleset(data.contact.geom)

    return True


def random_valid_config(
    rng: np.random.Generator,
    model: mujoco.MjModel,
    joints: list[str] = [],
    cr: CollisionRuleset | None = None,
    data: mujoco.MjData | None = None,
) -> np.ndarray:
    q_idx = qpos_idx(model, joints) if joints else list(range(model.nq))
    q_rand = rng.uniform(*model.jnt_range.T)[q_idx]
    while not is_valid_config(q_rand, model, q_idx, cr, data):
        q_rand = rng.uniform(*model.jnt_range.T)[q_idx]
    return q_rand


def shortcut(
    path: Path,
    model: mujoco.MjModel,
    cr: CollisionRuleset | None,
    validation_dist: float = 0.05,
    max_attempts: int = 100,
    seed: int | None = None,
) -> Path:
    """Perform shortcutting on a path.

    Args:
        path: The path to shortcut.
        cr: The CollisionRuleset to enforce (if any) for validation checks.
        validation_dist: The distance between each validation check, which occurs
            between a pair of waypoints that are trying to be directly connected
            if these waypoints are further than `validation_dist` apart.
        max_attempts: The maximum number of shortcut attempts. Each attempt
            will randomly select two waypoints in the path. If the path has
            exactly two waypoints, no more attempts will be executed.
        seed: The seed which is used for randomly picking pairs of waypoints
            to shortcut.

    Returns:
        A path with direct connections between each adjacent waypoint.
    """
    data = mujoco.MjData(model)
    data.qpos = path.q_init
    rng = np.random.default_rng(seed=seed)

    q_idx = qpos_idx(model, path.joints)

    # sanity check: can we shortcut directly between the start/end of the path?
    shortened_waypoints = _connect_waypoints(
        waypoints=path.waypoints,
        model=model,
        start_idx=0,
        end_idx=len(path.waypoints) - 1,
        validation_dist=validation_dist,
        q_idx=q_idx,
        data=data,
        cr=cr,
    )
    for _ in range(max_attempts):
        if len(shortened_waypoints) == 2:
            # we can go directly from start to goal, so no more shortcutting can be done
            return Path(
                q_init=path.q_init, waypoints=shortened_waypoints, joints=path.joints
            )
        # randomly pick 2 waypoints
        start, end = 0, 0
        while start == end:
            start, end = rng.integers(len(shortened_waypoints), size=2)
        if start > end:
            start, end = end, start
        shortened_waypoints = _connect_waypoints(
            waypoints=shortened_waypoints,
            model=model,
            start_idx=start,
            end_idx=end,
            validation_dist=validation_dist,
            q_idx=q_idx,
            data=data,
            cr=cr,
        )

    return Path(q_init=path.q_init, waypoints=shortened_waypoints, joints=path.joints)


def _connect_waypoints(
    waypoints: list[np.ndarray],
    model: mujoco.MjModel,
    start_idx: int,
    end_idx: int,
    validation_dist: float = 0.05,
    # TODO: re-visit this and see if it makes sense to have empty default
    q_idx: list[int] = [],
    data: mujoco.MjData | None = None,
    cr: CollisionRuleset | None = None,
) -> list[np.ndarray]:
    """If possible, directly connect two specific waypoints from a list of waypoints.

    Args:
        waypoints: The list of waypoints.
        start_idx: The index of the first waypoint.
        end_idx: The index of the second waypoint.
        validation_dist: The distance increment that is used for performing intermediate
            validation checks (see `jg`, `data`, and `cr`). This must be > 0.
        data: MuJoCo MjData. Used for validation checking.
            To disable validation checking, set this to None.
        cr: The CollisionRuleset to enforce (if any) for validation checks.

    Returns:
        A waypoint list with a direct connection between the waypoints at indices
        (`start_idx`, `end_idx`) if the waypoints at these indices can be connected.
    """
    if validation_dist <= 0.0:
        raise ValueError("`validation_dist` must be > 0")

    q_start = waypoints[start_idx]
    q_target = waypoints[end_idx]

    q_curr = step(q_start, q_target, validation_dist)
    while not np.array_equal(q_curr, q_target):
        if data is not None and not is_valid_config(q_curr, model, q_idx, cr, data):
            return waypoints
        q_curr = step(q_curr, q_target, validation_dist)
    return waypoints[: start_idx + 1] + waypoints[end_idx:]

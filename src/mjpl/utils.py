import mujoco
import numpy as np
from mink.lie import SE3, SO3

from .collision_ruleset import CollisionRuleset
from .joint_group import JointGroup


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


def site_pose(data: mujoco.MjData, site_name: str) -> SE3:
    """Get the pose of a site in the world frame.

    Args:
        data: MuJoCo data.
        site_name: The name of the site.

    Returns:
        The pose of the site in the world frame.
    """
    position = data.site(site_name).xpos.copy()
    rotation = data.site(site_name).xmat.copy()
    return SE3.from_rotation_and_translation(
        SO3.from_matrix(rotation.reshape(3, 3)),
        position,
    )


def is_valid_config(
    q: np.ndarray,
    jg: JointGroup,
    data: mujoco.MjData,
    cr: CollisionRuleset | None,
) -> bool:
    # Check joint limits.
    if not np.all((q >= jg.lower_limits) & (q <= jg.upper_limits)):
        return False

    if cr:
        # Check for collisions.
        # We have to run FK once data.qpos is updated before running the collision checker.
        jg.fk(q, data)
        mujoco.mj_collision(jg.model, data)
        return cr.obeys_ruleset(data.contact.geom)

    return True


def random_valid_config(
    rng: np.random.Generator,
    jg: JointGroup,
    data: mujoco.MjData,
    cr: CollisionRuleset | None,
) -> np.ndarray:
    q_rand = jg.random_config(rng)
    while not is_valid_config(q_rand, jg, data, cr):
        q_rand = jg.random_config(rng)
    return q_rand


def _connect_waypoints(
    path: list[np.ndarray],
    start_idx: int,
    end_idx: int,
    validation_dist: float = 0.05,
    jg: JointGroup | None = None,
    data: mujoco.MjData | None = None,
    cr: CollisionRuleset | None = None,
) -> list[np.ndarray]:
    """If possible, directly connect two specific waypoints from a path.

    Args:
        path: The path with waypoints to connect.
        start_idx: The index of the first waypoint.
        end_idx: The index of the second waypoint.
        validation_dist: The distance increment that is used for performing intermediate
            validation checks (see `jg`, `data`, and `cr`). This must be > 0.
        jg: The JointGroup to apply validation checks on.
            To disable validation checking, set this to None.
        data: MuJoCo MjData. Used for validation checking.
            To disable validation checking, set this to None.
        cr: The CollisionRuleset to enforce (if any) for validation checks.

    Returns:
        A path with a direct connection between the waypoints at indices
        (`start_idx`, `end_idx`) if the waypoints at these indices can be connected.
    """
    if validation_dist <= 0.0:
        raise ValueError("`validation_dist` must be > 0")

    if (jg is None and data is not None) or (jg is not None and data is None):
        raise ValueError("Both `jg` and `data` must either be None or not None.")
    validate = jg is not None and data is not None

    q_start = path[start_idx]
    q_target = path[end_idx]

    q_curr = step(q_start, q_target, validation_dist)
    while not np.array_equal(q_curr, q_target):
        if validate and not is_valid_config(q_curr, jg, data, cr):
            return path
        q_curr = step(q_curr, q_target, validation_dist)
    return path[: start_idx + 1] + path[end_idx:]


def shortcut(
    path: list[np.ndarray],
    jg: JointGroup,
    cr: CollisionRuleset | None,
    q_init: np.ndarray | None = None,
    validation_dist: float = 0.05,
    max_attempts: int = 100,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Perform shortcutting on a path.

    Args:
        path: The path to shortcut.
        jg: The JointGroup to apply validation checks on when shortcutting.
        cr: The CollisionRuleset to enforce (if any) for validation checks.
        q_init: Full initial configuration. Used to set values of joints that are
            not in `jg`.
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
    data = mujoco.MjData(jg.model)
    if q_init is not None:
        data.qpos = q_init
    rng = np.random.default_rng(seed=seed)

    # sanity check: can we shortcut directly between the start/end of the path?
    shortened_path = _connect_waypoints(
        path=path,
        start_idx=0,
        end_idx=len(path) - 1,
        validation_dist=validation_dist,
        jg=jg,
        data=data,
        cr=cr,
    )
    for _ in range(max_attempts):
        if len(shortened_path) == 2:
            # we can go directly from start to goal, so no more shortcutting can be done
            return shortened_path
        # randomly pick 2 waypoints
        start, end = 0, 0
        while start == end:
            start, end = rng.integers(len(shortened_path), size=2)
        if start > end:
            start, end = end, start
        shortened_path = _connect_waypoints(
            path=shortened_path,
            start_idx=start,
            end_idx=end,
            validation_dist=validation_dist,
            jg=jg,
            data=data,
            cr=cr,
        )

    return shortened_path

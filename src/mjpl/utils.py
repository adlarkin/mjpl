import mujoco
import numpy as np
from mink.lie import SE3, SO3

from .collision_ruleset import CollisionRuleset
from .joint_group import JointGroup


def configuration_distance(q_from: np.ndarray, q_to: np.ndarray):
    return np.linalg.norm(q_to - q_from)


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
    fill: bool,
    min_fill_dist: float = 0.05,
    jg: JointGroup | None = None,
    data: mujoco.MjData | None = None,
    cr: CollisionRuleset | None = None,
) -> list[np.ndarray]:
    """If possible, directly connect two specific waypoints from a path.

    This is meant to be a helper function for things like `shortcut` and
    `fill_path`.

    Args:
        path: The path with waypoints to connect.
        start_idx: The index of the first waypoint.
        end_idx: The index of the second waypoint.
        fill: Whether or not filler waypoints should be added in between the
            `start_idx` and `end_idx` waypoints (if they can be directly
            connected).
        min_fill_dist: The distance increment that is used for adding filler
            waypoints and performing intermediate validation checks (see `jg`,
            `data`, and `cr`). This must be > 0.
        jg: The JointGroup to apply validation checks on.
            To disable validation checking, set this to None.
        data: MuJoCo MjData. Used for validation checking.
            To disable validation checking, set this to None.
        cr: The CollisionRuleset to enforce (if any) for validation checks.

    Returns:
        A path with a direct connection between the waypoints at indices
        (`start_idx`, `end_idx`) with optional intermediate waypoints if the waypoints
        at these indices can be connected.

    Notes:
        `jg` and `data` must either be None (which disables validation checks)
        or not None (which enables validation checks).
    """
    if min_fill_dist <= 0.0:
        raise ValueError("`min_fill_dist` must be > 0")

    variables = [jg, data]
    if any(var is not None for var in variables) and any(
        var is None for var in variables
    ):
        raise ValueError("Both `jg` and `data` must either be None or not None.")
    validate = all(var is not None for var in variables)

    q_start_idx = path[start_idx]
    q_target = path[end_idx]

    intermediate_waypoints = []
    q_curr = step(q_start_idx, q_target, min_fill_dist)
    while not np.array_equal(q_curr, q_target):
        if validate and not is_valid_config(q_curr, jg, data, cr):
            return path
        if fill and not np.array_equal(q_curr, q_target):
            intermediate_waypoints.append(q_curr.copy())
        q_curr = step(q_curr, q_target, min_fill_dist)
    return path[: start_idx + 1] + intermediate_waypoints + path[end_idx:]


def shortcut(
    path: list[np.ndarray],
    jg: JointGroup,
    model: mujoco.MjModel,
    cr: CollisionRuleset | None,
    validation_dist: float = 0.05,
    max_attempts: int = 100,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Perform shortcutting on a path.

    Args:
        path: The path to shortcut.
        jg: The JointGroup to apply validation checks on when shortcutting.
        model: MuJoCo model.
        cr: The CollisionRuleset to enforce (if any) for validation checks.
        validation_dist: The distance between each validation check,
            which occurs between a pair of waypoints that are trying to be
            directly connected if these waypoints are further than
            `validation_dist` apart.
        max_attempts: The maximum number of shortcut attempts. Each attempt
            will randomly select two waypoints in the path. If the path has
            exactly two waypoints, no more attempts will be executed.
        seed: The seed which is used for randomly picking pairs of waypoints
            to shortcut.

    Returns:
        A path with direct connections between each adjacent waypoint.

    Notes:
        Shortuctting does not perform waypoint filling. See `fill_path`.
    """
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed=seed)

    # sanity check: can we shortcut directly between the start/end of the path?
    shortened_path = _connect_waypoints(
        path=path,
        start_idx=0,
        end_idx=len(path) - 1,
        fill=False,
        min_fill_dist=validation_dist,
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
            fill=False,
            min_fill_dist=validation_dist,
            jg=jg,
            data=data,
            cr=cr,
        )

    return shortened_path


# TODO: consider removing this method if it's not needed
def fill_path(
    path: list[np.ndarray], max_dist_between_points: float
) -> list[np.ndarray]:
    """Perform waypoint filling on a path.

    Args:
        path: The path to fill.
        max_dist_between_points: The distance between filler waypoints.

    Returns:
        A path with intermediate waypoints between adjacent waypoints in
        `path` that are further than `max_dist_between_points` apart.
    """
    filled_path = [path[0]]
    for i in range(len(path) - 1):
        filled_segment = _connect_waypoints(
            path=[path[i], path[i + 1]],
            start_idx=0,
            end_idx=1,
            fill=True,
            min_fill_dist=max_dist_between_points,
        )
        filled_path.extend(filled_segment[1:])

    return filled_path

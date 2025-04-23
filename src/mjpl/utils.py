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


def qpos_idx(
    model: mujoco.MjModel, joints: list[str], default_to_full: bool = False
) -> list[int]:
    """Get the indices in mujoco.MjData.qpos that correspond to specific joints.

    Args:
        model: MuJoCo model.
        joints: The names of the joints in `model`.
        default_to_full: Whether or not all indices in mujoco.MjData.qpos should be
            returned if `joints` is an empty list.

    Returns:
        A list of indices that correspond to `joints` in mujoco.MjData.qpos.
    """
    if not joints and default_to_full:
        return list(range(model.nq))

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
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cr: CollisionRuleset | None,
) -> bool:
    """Check if the configuration stored in MjData.qpos is valid.

    "Valid" in this case means joint limits and an (optional) CollisionRuleset
    are not violated.

    Args:
        model: MuJoCo model.
        data: MuJoCo data, which has the configuration to validate.
        cr: CollisionRuleset the configuration must obey. Set this to None if no
            CollisionRuleset should be enforced.

    Returns:
        True if the configuration in `data` does not violate joint limits and obeys
        `cr` (if `cr` is defined). False otherwise.
    """
    # Check joint limits.
    if not np.all(
        (data.qpos >= model.jnt_range[:, 0]) & (data.qpos <= model.jnt_range[:, 1])
    ):
        return False

    if cr:
        # Check for collisions.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        return cr.obeys_ruleset(data.contact.geom)

    return True


def random_valid_config(
    model: mujoco.MjModel,
    q_init: np.ndarray,
    seed: int | None = None,
    joints: list[str] = [],
    cr: CollisionRuleset | None = None,
) -> np.ndarray:
    """Generate a random valid configuration.

    See `is_valid_config` for notes about "validity" of a configration.

    Args:
        model: MuJoCo model.
        q_init: Initial joint configuration. Used to set values for joints that are
            not in `joints`.
        seed: Seed used for random number generation.
        joints: The joints to set random values for. Set this to an empty list if all
            joints should be set randomly.
        cr: CollisionRuleset the randomly generated configuration must obey. Set this
            to None if no CollisionRuleset should be enforced.

    Returns:
        A random valid configuration.
    """
    data = mujoco.MjData(model)
    data.qpos = q_init

    rng = np.random.default_rng(seed=seed)
    q_idx = qpos_idx(model, joints, default_to_full=True)
    data.qpos[q_idx] = rng.uniform(*model.jnt_range.T)[q_idx]
    while not is_valid_config(model, data, cr):
        data.qpos[q_idx] = rng.uniform(*model.jnt_range.T)[q_idx]

    return data.qpos


def shortcut(
    model: mujoco.MjModel,
    path: Path,
    cr: CollisionRuleset | None,
    validation_dist: float = 0.05,
    max_attempts: int = 100,
    seed: int | None = None,
) -> Path:
    """Perform shortcutting on a path.

    Args:
        model: MuJoCo model.
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

    q_idx = qpos_idx(model, path.joints, default_to_full=True)

    # sanity check: can we shortcut directly between the start/end of the path?
    shortened_waypoints = _connect_waypoints(
        model,
        data,
        path.waypoints,
        q_idx,
        start_idx=0,
        end_idx=len(path.waypoints) - 1,
        validation_dist=validation_dist,
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
            model,
            data,
            shortened_waypoints,
            q_idx,
            start_idx=start,
            end_idx=end,
            validation_dist=validation_dist,
            cr=cr,
        )

    return Path(q_init=path.q_init, waypoints=shortened_waypoints, joints=path.joints)


def _connect_waypoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    waypoints: list[np.ndarray],
    q_idx: list[int],
    start_idx: int,
    end_idx: int,
    validation_dist: float,
    cr: CollisionRuleset | None,
) -> list[np.ndarray]:
    """If possible, directly connect two specific waypoints from a list of waypoints.

    Args:
        model: MuJoCo model.
        data: MuJoCo data. Used for validation checking. This should have values
            initialized in MjData.qpos that do not correspond to `waypoints`/`q_idx`.
        waypoints: The list of waypoints.
        q_idx: The indices in MjData.qpos that correspond to the arrays in `waypoints`.
        start_idx: The index of the first waypoint.
        end_idx: The index of the second waypoint.
        validation_dist: The distance increment used for performing intermediate
            validation checks. This must be > 0.
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
        data.qpos[q_idx] = q_curr
        if not is_valid_config(model, data, cr):
            return waypoints
        q_curr = step(q_curr, q_target, validation_dist)
    return waypoints[: start_idx + 1] + waypoints[end_idx:]

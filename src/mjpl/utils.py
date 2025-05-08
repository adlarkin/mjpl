import mink
import mujoco
import numpy as np

from .constraint.constraint_interface import Constraint
from .constraint.utils import apply_constraints


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

    # TODO: revisit this. Here (and a few lines below) I am using q_init for q_old
    # to obey API changes (I think that's actually ok) - if I stick with this,
    # maybe make a note of it in docs above for the `q_init` arg?
    q_constrained = apply_constraints(q_init, q, constraints)
    while q_constrained is None:
        q[q_idx] = rng.uniform(*model.jnt_range.T)[q_idx]
        q_constrained = apply_constraints(q_init, q, constraints)
    return q_constrained


def path_length(waypoints: list[np.ndarray]) -> float:
    """Compute the path length in configuration space.

    Args:
        waypoints: A list of waypoints that form the path.

    Returns:
        The length of the waypoint list in configuration space.
    """
    path = np.array(waypoints)
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(segment_lengths)

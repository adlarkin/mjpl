from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

import mujoco
import numpy as np

from ..collision_ruleset import CollisionRuleset
from ..joint_group import JointGroup


@dataclass(frozen=True)
class Trajectory:
    """Trajectory data.

    `n` is the number of states in the trajectory.
    The trajectory duration (with final time t_f) is `dt` * n.
    """

    # The timestep between each position, velocity, and acceleration snapshot.
    dt: float
    # Position snapshots at increments of dt, ranging from t = [dt, t_f].
    positions: list[np.ndarray]
    # Velocity snapshots at increments of dt, ranging from t = [dt, t_f]
    velocities: list[np.ndarray]
    # Acceleration snapshots at increments of dt, ranging from t = [dt, t_f]
    accelerations: list[np.ndarray]


class TrajectoryGenerator(ABC):
    """Abstract base class for generating trajectories."""

    @final
    def generate_trajectory(
        self,
        path: list[np.ndarray],
        q_init: np.ndarray | None = None,
        jg: JointGroup | None = None,
        cr: CollisionRuleset | None = None,
    ) -> Trajectory:
        """Generate a trajectory that follows a path and (optionally) obeys a collision ruleset.

        It is assumed that straight-line connections between adjacent waypoints in the
        path obey the collision ruleset. The process to ensure a trajectory obeys the
        collision ruleset is as follows:
            1. Generate a trajectory.
            2. If part of the trajectory violates the collision ruleset, add an
               intermediate waypoint to the path segment that corresponds to the part
               of the trajectory that violates the collision ruleset.
            3. Repeat steps 1-2 until the trajectory has no segments that violate the
               collision ruleset.

        This is taken from section 3.5 of https://groups.csail.mit.edu/rrg/papers/Richter_ISRR13.pdf

        Args:
            path: The path the trajectory must follow.
            generator: Trajectory generator.
            q_init: The initial joint configuration of the world. This should only be
                defined if the trajectory must obey a collision ruleset.
            jg: The joints used for the path and trajectory. This should only be
                defined if the trajectory must obey a collision ruleset.
            cr: The collision ruleset the trajectory must adhere to. This should only be
                defined if the trajectory must obey a collision ruleset.

        Returns:
            A trajectory that follows `path` and (optionally) obeys `cr`.
        """
        if q_init is None and jg is None and cr is None:
            return self._build(path)
        elif q_init is not None and jg is not None and cr is not None:
            data = mujoco.MjData(jg.model)
            data.qpos = q_init
            filled_path = path.copy()
            while True:
                traj = self._build(filled_path)
                for q in traj.positions:
                    jg.fk(q, data)
                    mujoco.mj_collision(jg.model, data)
                    if not cr.obeys_ruleset(data.contact.geom):
                        # Add an intermediate waypoint to the section of the path that
                        # corresponds to the trajectory position that's in collision.
                        segment_start, segment_end = _closest_segment(q, filled_path)
                        intermediate_waypoint = (
                            filled_path[segment_start] + filled_path[segment_end]
                        ) / 2
                        filled_path.insert(segment_end, intermediate_waypoint)
                        break
                else:
                    return traj
        else:
            raise ValueError("`q_init`, `jg`, and `cr` must all be defined or `None`.")

    @abstractmethod
    def _build(self, path: list[np.ndarray]) -> Trajectory:
        """Build a trajectory that follows a path.

        Args:
            path: A sequence of waypoints the trajectory should follow.

        Returns:
            A trajectory that follows `path`.
        """
        pass


def _closest_segment(waypoint: np.ndarray, path: list[np.ndarray]) -> tuple[int, int]:
    """Find the segment of a path that corresponds to an arbitrary waypoint.

    Args:
        waypoint: The waypoint.
        path: The path.

    Returns:
        The [start,end] indices of `path` that correspond to `waypoint`.
    """
    min_dist = float("inf")
    best_idx = -1

    for i in range(len(path) - 1):
        p0 = path[i]
        p1 = path[i + 1]
        seg_vec = p1 - p0
        pt_vec = waypoint - p0

        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq == 0:
            continue

        # Project point onto the segment, clamp to [0, 1].
        projection = np.dot(pt_vec, seg_vec) / seg_len_sq
        alpha = np.clip(projection, 0.0, 1.0)
        closest = p0 + alpha * seg_vec

        dist = np.linalg.norm(waypoint - closest)
        if dist < min_dist:
            min_dist = dist
            best_idx = i

    return best_idx, best_idx + 1

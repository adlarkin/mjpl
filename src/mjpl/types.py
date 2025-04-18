from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Path:
    # Initial configuration of all joints.
    q_init: np.ndarray
    # Configurations for `joints` that define a path, starting at `q_init`.
    waypoints: list[np.ndarray]
    # The joints corresponding to `waypoints`.
    # None means all joints are used.
    joints: list[str] | None

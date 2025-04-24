from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Path:
    """Path data."""

    # Initial configuration of all joints.
    q_init: np.ndarray
    # Configurations for `joints` that define a path, starting at `q_init`.
    waypoints: list[np.ndarray]
    # The joints corresponding to `waypoints`.
    joints: list[str]

    def __post_init__(self) -> None:
        if not self.joints:
            raise ValueError("`joints` cannot be empty.")

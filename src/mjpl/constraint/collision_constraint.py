import mujoco
import numpy as np

from ..collision_ruleset import CollisionRuleset
from .constraint_interface import Constraint


class CollisionConstraint(Constraint):
    """Constraint that enforces a CollisionRuleset on a configuration."""

    def __init__(self, model: mujoco.MjModel, cr: CollisionRuleset) -> None:
        """Constructor.

        Args:
            model: MuJoCo model.
            cr: CollisionRuleset that is checked against a configuration.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.cr = cr

    def valid_config(self, q: np.ndarray) -> bool:
        self.data.qpos = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        return self.cr.obeys_ruleset(self.data.contact.geom)

    def apply(self, q: np.ndarray) -> np.ndarray | None:
        return q if self.valid_config(q) else None

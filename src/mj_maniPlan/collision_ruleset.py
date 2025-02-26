import mujoco
import numpy as np


# NOTE: allowed_collisions are BODY ids, collision_matrix are GEOM ids
class CollisionRuleset:
    """Class that defines which bodies are allowed to be in collision.

    This can be used with the contact information in MjData.
    """

    def __init__(
        self, model: mujoco.MjModel, allowed_collision_bodies: list[tuple[int, int]]
    ) -> None:
        """Constructor.

        Args:
            model: MuJoCo model.
            allowed_collision_bodies: Bodies that are allowed to be in collision.
        """
        self.model = model
        self.allowed_collision_bodies = [
            set(body_pair) for body_pair in allowed_collision_bodies
        ]

    def obeys_ruleset(self, collision_matrix: np.ndarray) -> bool:
        """Check if a collision matrix violates the allowed body collisions.

        A collision matrix defines geometries that are in collision.

        Args:
            collision_matrix: A nx2 matrix, where n=number of collisions. Each
                              row represents a pair of geometries that are in collision.

        Returns:
            True if the geometry pairs in the collision matrix map to allowed body
            collision pairs. False otherwise.
        """
        # collision matrix should be n x 2, where n = number of collisions
        # (each row in the collision matrix represents a pair of geoms that are in collision)
        #
        # When working with MjData, collision matrix is stored in data.contact.geom
        assert collision_matrix.shape[1] == 2
        for geom1, geom2 in collision_matrix:
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            if {body1, body2} not in self.allowed_collision_bodies:
                return False
        return True

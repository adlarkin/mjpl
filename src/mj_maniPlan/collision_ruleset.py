import mujoco
import numpy as np


# NOTE: allowed_collisions are BODY ids, collision_matrix are GEOM ids
class CollisionRuleset:
    """Class that defines which bodies are allowed to be in collision.

    This can be used with the contact information in MjData.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        allowed_collision_bodies: np.ndarray = np.empty((0, 2)),
    ) -> None:
        """Constructor.

        Args:
            model: MuJoCo model.
            allowed_collision_bodies: Bodies that are allowed to be in collision.
                                      This should be a nx2 matrix, where each row
                                      specifies a pair of bodies that are allowed
                                      to be in collision.
        """
        self.model = model
        assert allowed_collision_bodies.shape[1] == 2
        self._allowed_collision_bodies = allowed_collision_bodies
        # Convert allowed body matrix to a set of tuples. Sort matrix rows
        # since equivalence between two collision ID pairs is order independent.
        self.allowed_set = {
            tuple(row) for row in np.sort(allowed_collision_bodies, axis=1)
        }

    def obeys_ruleset(self, collision_matrix: np.ndarray) -> bool:
        """Check if a collision matrix violates the allowed body collisions.

        A collision matrix defines geometries that are in collision.

        In MjData, the collision matrix is stored in data.contact.geom

        Args:
            collision_matrix: A nx2 matrix, where n=number of collisions. Each row
                              represents a pair of geometries that are in collision.

        Returns:
            True if the geometry pairs in the collision matrix map to allowed body
            collision pairs. False otherwise.
        """
        assert collision_matrix.shape[1] == 2
        if collision_matrix.shape[0] == 0:
            # No collisions
            return True
        elif self._allowed_collision_bodies.shape[0] == 0:
            # Collisions are present, but the ruleset doesn't allow any collisions
            return False
        # Map geometry IDs to their respective body IDs. Sort matrix rows since
        # equivalence between two collision ID pairs is order independent.
        collision_bodies = self.model.geom_bodyid[np.sort(collision_matrix, axis=1)]
        # Convert the collision matrix to a set of tuples.
        # The collision matrix obeys the ruleset if the collision body set is
        # a subset of the allowed body set.
        return all(tuple(row) in self.allowed_set for row in collision_bodies)

    @property
    def allowed_collision_bodies(self) -> np.ndarray:
        """The matrix of allowed collision bodies."""
        return self._allowed_collision_bodies.copy()

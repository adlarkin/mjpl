import mujoco
import numpy as np


class CollisionRuleset:
    """Class that defines which bodies are allowed to be in collision.

    This can be used with the contact information in MjData.
    """

    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        allowed_collision_bodies: list[tuple[str, str]] = [],
    ) -> None:
        """Constructor.

        Args:
            model: MuJoCo model. If no bodies are allowed to be in collision, this
                can be set to None.
            allowed_collision_bodies: List of body pairs that are allowed to be in
                collision. An empty list means no bodies are allowed to be in collision.
        """
        if allowed_collision_bodies and model is None:
            raise ValueError(
                "`model` must be defined if `allowed_collision_bodies` is not empty."
            )

        self.model = model
        self.allowed_collisions = None
        if allowed_collision_bodies:
            # Create a sorted body ID allowed collision matrix.
            # Sort since collisions between bodies a and b can be represented as
            # (a,b) or (b,a).
            body_ids = [
                (self.model.body(a).id, self.model.body(b).id)
                for a, b in allowed_collision_bodies
            ]
            self.allowed_collisions = np.sort(body_ids, axis=1)

    def obeys_ruleset(self, collision_geometries: np.ndarray) -> bool:
        """Check if a collision matrix adheres to the allowed body collisions.

        A collision matrix defines geometries that are in collision.
        In MuJoCo, the collision matrix is stored in MjData.contact.geom

        Args:
            collision_geometries: A nx2 matrix, where n=number of collisions. Each row
                is a pair of geometry IDs that are in collision.

        Returns:
            True if all geometry pairs in the collision matrix map to allowed body
            collision pairs. False otherwise.
        """
        if collision_geometries.shape[1] != 2:
            raise ValueError("`collision_geometries` must be a nx2 matrix.")

        if collision_geometries.shape[0] == 0:
            # No collisions
            return True
        elif self.allowed_collisions is None:
            # Collisions are present, but the ruleset doesn't allow any collisions
            return False

        # Map geometry IDs to their respective body IDs, and then check if all body
        # collisions are part of the allowed collision matrix.
        # Sort since collisions between bodies a and b can be represented as
        # (a,b) or (b,a).
        collision_bodies = np.sort(self.model.geom_bodyid[collision_geometries], axis=1)
        matches = (
            collision_bodies[:, None, :] == self.allowed_collisions[None, :, :]
        ).all(axis=2)
        return np.all(matches.any(axis=1))

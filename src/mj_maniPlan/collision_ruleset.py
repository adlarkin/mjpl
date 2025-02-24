import mujoco


class CollisionRuleset:
    def __init__(
        self, model: mujoco.MjModel, allowed_collisions: list[tuple[int, int]]
    ) -> None:
        self.model = model
        self.allowed_collisions = [set(body_pair) for body_pair in allowed_collisions]

    def obeys_ruleset(self, data: mujoco.MjData) -> bool:
        for geom1, geom2 in data.contact.geom:
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            if {body1, body2} not in self.allowed_collisions:
                return False
        return True

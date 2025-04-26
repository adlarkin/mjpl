import unittest

import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl


class TestCollisionRuleset(unittest.TestCase):
    def setUp(self):
        self.model = load_robot_description("panda_mj_description")

        # Geometry ids that can be used for creating a collision matrix.
        # A body may have multiple geometries associated with it.
        # Here, we are using the first geometry tied to a body.
        # To get all geometries tied to a body, see:
        # https://github.com/kevinzakka/mink/blob/cce9cf4ed13e461dc1b3d38fe88f245700aa98c2/mink/utils.py#L137
        self.link1_geom = self.model.body_geomadr[self.model.body("link1").id]
        self.link2_geom = self.model.body_geomadr[self.model.body("link2").id]
        self.link3_geom = self.model.body_geomadr[self.model.body("link3").id]
        self.link4_geom = self.model.body_geomadr[self.model.body("link4").id]
        self.link5_geom = self.model.body_geomadr[self.model.body("link5").id]
        self.link6_geom = self.model.body_geomadr[self.model.body("link6").id]
        self.link7_geom = self.model.body_geomadr[self.model.body("link7").id]
        self.left_finger_geom = self.model.body_geomadr[
            self.model.body("left_finger").id
        ]
        self.right_finger_geom = self.model.body_geomadr[
            self.model.body("right_finger").id
        ]

    def test_single_valid_collision(self):
        valid_collision_bodies = [("link1", "link2")]
        cr = mjpl.CollisionRuleset(self.model, valid_collision_bodies)

        geom_collision_matrix = np.empty((0, 2))
        self.assertTrue(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.link1_geom, self.link2_geom],
            ]
        )
        self.assertTrue(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.link2_geom, self.link1_geom],
            ]
        )
        self.assertTrue(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.link1_geom, self.link2_geom],
                [self.link2_geom, self.link1_geom],
            ]
        )
        self.assertTrue(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.left_finger_geom, self.right_finger_geom],
            ]
        )
        self.assertFalse(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.link1_geom, self.link5_geom],
            ]
        )
        self.assertFalse(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.link1_geom, self.link2_geom],
                [self.link1_geom, self.link5_geom],
            ]
        )
        self.assertFalse(cr.obeys_ruleset(geom_collision_matrix))

    def test_multiple_valid_collisions(self):
        valid_collision_bodies = [
            ("link1", "link2"),
            ("link6", "link7"),
            ("left_finger", "right_finger"),
        ]
        cr = mjpl.CollisionRuleset(self.model, valid_collision_bodies)

        geom_collision_matrix = np.empty((0, 2))
        self.assertTrue(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.link1_geom, self.link2_geom],
            ]
        )
        self.assertTrue(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.link2_geom, self.link1_geom],
            ]
        )
        self.assertTrue(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.link6_geom, self.link7_geom],
            ]
        )
        self.assertTrue(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.left_finger_geom, self.right_finger_geom],
                [self.link7_geom, self.link6_geom],
                [self.link1_geom, self.link2_geom],
                [self.link2_geom, self.link1_geom],
            ]
        )
        self.assertTrue(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.link2_geom, self.link3_geom],
            ]
        )
        self.assertFalse(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.link4_geom, self.link5_geom],
            ]
        )
        self.assertFalse(cr.obeys_ruleset(geom_collision_matrix))

        geom_collision_matrix = np.array(
            [
                [self.left_finger_geom, self.right_finger_geom],
                [self.link1_geom, self.right_finger_geom],
                [self.link7_geom, self.link6_geom],
            ]
        )
        self.assertFalse(cr.obeys_ruleset(geom_collision_matrix))

    def test_no_valid_collisions(self):
        cr = mjpl.CollisionRuleset()

        no_collisions = np.empty((0, 2))
        self.assertTrue(cr.obeys_ruleset(no_collisions))

        geom_collision_matrix = np.array(
            [
                [self.link3_geom, self.link2_geom],
            ]
        )
        self.assertFalse(cr.obeys_ruleset(geom_collision_matrix))

    def test_invalid_args(self):
        with self.assertRaisesRegex(ValueError, "must be defined"):
            mjpl.CollisionRuleset(
                model=None, allowed_collision_bodies=[("body1", "body2")]
            )

        cr = mjpl.CollisionRuleset()
        with self.assertRaisesRegex(ValueError, "nx2"):
            cr.obeys_ruleset(np.zeros((1, 3)))
            cr.obeys_ruleset(np.zeros((1, 2, 1)))


if __name__ == "__main__":
    unittest.main()

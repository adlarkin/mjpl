import unittest

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mj_maniPlan as mjpl


class TestCollisionRuleset(unittest.TestCase):
    def setUp(self):
        self.model = load_robot_description("panda_mj_description")
        self.data = mujoco.MjData(self.model)

        # Body ids that can be used for defining valid collisions
        self.link1 = self.model.body("link1").id
        self.link2 = self.model.body("link2").id
        self.link3 = self.model.body("link3").id
        self.link4 = self.model.body("link4").id
        self.link5 = self.model.body("link5").id
        self.link6 = self.model.body("link6").id
        self.link7 = self.model.body("link7").id
        self.left_finger = self.model.body("left_finger").id
        self.right_finger = self.model.body("right_finger").id

        # Geometry ids that can be used for creating a collision matrix.
        # A body may have multiple geometries associated with it.
        # Here, we are using the first geometry tied to a body.
        # To get all geometries tied to a body, see:
        # https://github.com/kevinzakka/mink/blob/cce9cf4ed13e461dc1b3d38fe88f245700aa98c2/mink/utils.py#L137
        self.link1_geom = self.model.body_geomadr[self.link1]
        self.link2_geom = self.model.body_geomadr[self.link2]
        self.link3_geom = self.model.body_geomadr[self.link3]
        self.link4_geom = self.model.body_geomadr[self.link4]
        self.link5_geom = self.model.body_geomadr[self.link5]
        self.link6_geom = self.model.body_geomadr[self.link6]
        self.link7_geom = self.model.body_geomadr[self.link7]
        self.left_finger_geom = self.model.body_geomadr[self.left_finger]
        self.right_finger_geom = self.model.body_geomadr[self.right_finger]

    def test_single_valid_collision(self):
        valid_collision_bodies = np.array(
            [
                [self.link1, self.link2],
            ]
        )
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
        valid_collision_bodies = np.array(
            [
                [self.link1, self.link2],
                [self.link6, self.link7],
                [self.left_finger, self.right_finger],
            ]
        )
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
        cr = mjpl.CollisionRuleset(self.model)

        no_collisions = np.empty((0, 2))
        self.assertTrue(cr.obeys_ruleset(no_collisions))

        geom_collision_matrix = np.array(
            [
                [self.link3_geom, self.link2_geom],
            ]
        )
        self.assertFalse(cr.obeys_ruleset(geom_collision_matrix))

    def test_get_allowed_collision_bodies(self):
        valid_collision_bodies = np.array(
            [
                [self.link3, self.link4],
                [self.link5, self.link6],
            ]
        )
        cr = mjpl.CollisionRuleset(self.model, valid_collision_bodies)

        allowed_bodies = cr.allowed_collision_bodies
        self.assertTrue(np.array_equal(allowed_bodies, valid_collision_bodies))
        self.assertIsNot(allowed_bodies, valid_collision_bodies)


if __name__ == "__main__":
    unittest.main()

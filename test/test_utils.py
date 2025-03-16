import unittest
from pathlib import Path

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mj_maniPlan.utils as utils
from mj_maniPlan.collision_ruleset import CollisionRuleset
from mj_maniPlan.joint_group import JointGroup

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_BALL_XY_PLANE_XML = _MODEL_DIR / "ball_xy_plane.xml"


class TestUtils(unittest.TestCase):
    def test_step(self):
        start = np.array([0.0, 0.0])
        target = np.array([0.5, 0.0])

        q_next = utils.step(start, target, max_step_dist=5.0)
        np.testing.assert_equal(q_next, target)

        q_next = utils.step(start, target, max_step_dist=0.1)
        np.testing.assert_allclose(q_next, np.array([0.1, 0.0]), rtol=0, atol=1e-8)

        with self.assertRaisesRegex(ValueError, "`max_step_dist` must be > 0.0"):
            utils.step(start, target, max_step_dist=0.0)
            utils.step(start, target, max_step_dist=-1.0)

    def test_connect_waypoints(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XY_PLANE_XML.as_posix())
        planning_joints = [
            model.joint("ball_slide_x").id,
            model.joint("ball_slide_y").id,
        ]
        jg = JointGroup(model, planning_joints)
        cr = CollisionRuleset(model)

        path = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0]),
            # This is outside of the model's joint limits
            np.array([3.0, 0.0]),
        ]

        # connect with validation checking and no filling
        connected_path = utils._connect_waypoints(
            path,
            start_idx=0,
            end_idx=2,
            fill=False,
            min_fill_dist=0.7,
            jg=jg,
            data=mujoco.MjData(model),
            cr=cr,
        )
        self.assertEqual(len(connected_path), 3)
        np.testing.assert_equal(connected_path[0], path[0])
        np.testing.assert_equal(connected_path[1], path[2])
        np.testing.assert_equal(connected_path[2], path[3])

        # connect with validation checking and filling
        connected_path = utils._connect_waypoints(
            path,
            start_idx=0,
            end_idx=2,
            fill=True,
            min_fill_dist=0.7,
            jg=jg,
            data=mujoco.MjData(model),
            cr=cr,
        )
        self.assertEqual(len(connected_path), 5)
        np.testing.assert_equal(connected_path[0], path[0])
        np.testing.assert_allclose(
            connected_path[1], np.array([0.7, 0.0]), rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            connected_path[2], np.array([1.4, 0.0]), rtol=0, atol=1e-8
        )
        np.testing.assert_equal(connected_path[3], path[2])
        np.testing.assert_equal(connected_path[4], path[3])

        # connect with validation checking. Since the `end` waypoint idx
        # corresponds to a waypoint that violates the joint limits, this
        # should fail (i.e., the returned path is an unmodified copy of
        # the original)
        connected_path = utils._connect_waypoints(
            path,
            start_idx=1,
            end_idx=3,
            fill=True,
            min_fill_dist=0.7,
            jg=jg,
            data=mujoco.MjData(model),
            cr=cr,
        )
        self.assertEqual(len(connected_path), len(path))
        np.testing.assert_equal(connected_path[0], path[0])
        np.testing.assert_equal(connected_path[1], path[1])
        np.testing.assert_equal(connected_path[2], path[2])
        np.testing.assert_equal(connected_path[3], path[3])

        # connect without validation checking on the waypoint that violates
        # joint limits
        connected_path = utils._connect_waypoints(
            path,
            start_idx=1,
            end_idx=3,
            fill=False,
            min_fill_dist=0.1,
            jg=None,
            data=None,
        )
        self.assertEqual(len(connected_path), 3)
        np.testing.assert_equal(connected_path[0], path[0])
        np.testing.assert_equal(connected_path[1], path[1])
        np.testing.assert_equal(connected_path[2], path[3])

        with self.assertRaisesRegex(
            ValueError, "`jg` and `data` must either be None or not None"
        ):
            utils._connect_waypoints(
                path,
                start_idx=0,
                end_idx=2,
                fill=False,
                jg=None,
                data=mujoco.MjData(model),
            )
            utils._connect_waypoints(
                path, start_idx=0, end_idx=2, fill=False, jg=jg, data=None
            )

        with self.assertRaisesRegex(ValueError, "`min_fill_dist` must be > 0"):
            utils._connect_waypoints(
                path, start_idx=0, end_idx=2, fill=False, min_fill_dist=0.0
            )
            utils._connect_waypoints(
                path, start_idx=0, end_idx=2, fill=False, min_fill_dist=-1.0
            )

    def test_shortcut(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XY_PLANE_XML.as_posix())
        planning_joints = [
            model.joint("ball_slide_x").id,
            model.joint("ball_slide_y").id,
        ]
        jg = JointGroup(model, planning_joints)
        cr = CollisionRuleset(model)

        """
        Path that can benefit from shortcutting.

                p2 -> p3 -> p4 -> p5
                ^                 |
                |                 |
                p1 <- p0          |
                                  |
                                  v
                                  p6
        """
        p0 = np.array([0.0, 0.0])
        p1 = np.array([-0.1, 0.0])
        p2 = np.array([-0.1, 0.1])
        p3 = np.array([0.0, 0.1])
        p4 = np.array([0.1, 0.1])
        p5 = np.array([0.2, 0.1])
        p6 = np.array([0.2, -0.1])
        path = [p0, p1, p2, p3, p4, p5, p6]

        # all waypoints in the path are valid and directly connectable, so this
        # should result in a direct connection between path[0] and path[-1]
        shortened_path = utils.shortcut(path, jg, model, cr, seed=5)
        self.assertTrue(len(shortened_path), 2)
        np.testing.assert_equal(shortened_path[0], path[0])
        np.testing.assert_equal(shortened_path[1], path[-1])

        # Add a final point to the path that violates joint limits.
        # This means that after enough tries, the first and penultimate
        # (i.e., last valid) waypoints can be directly connected
        invalid_path = path + [np.array([3.0, -5.0])]
        shortened_path = utils.shortcut(
            invalid_path, jg, model, cr, max_attempts=20, seed=42
        )
        self.assertTrue(len(shortened_path), 3)
        np.testing.assert_equal(shortened_path[0], invalid_path[0])
        np.testing.assert_equal(shortened_path[1], invalid_path[-2])
        np.testing.assert_equal(shortened_path[2], invalid_path[-1])

    def test_shortcut_6dof(self):
        model = load_robot_description("ur5e_mj_description")

        data = mujoco.MjData(model)
        arm_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        arm_joint_ids = [model.joint(joint).id for joint in arm_joints]
        jg = JointGroup(model, arm_joint_ids)
        cr = CollisionRuleset(model)

        seed = 42

        # Make a path starting from the home config that connects to various random valid configs.
        path = [model.keyframe("home").qpos.copy()]
        rng = np.random.default_rng(seed=seed)
        random_waypoints = [
            utils.random_valid_config(rng, jg, data, cr) for _ in range(5)
        ]
        path.extend(random_waypoints)

        # Perform shortcutting. The path should now be shorter, but still start
        # and end at the same waypoints.
        shortcut_path = utils.shortcut(path, jg, model, cr, seed=seed)
        self.assertLess(len(shortcut_path), len(path))
        self.assertGreaterEqual(len(shortcut_path), 2)
        np.testing.assert_equal(shortcut_path[0], path[0])
        np.testing.assert_equal(shortcut_path[-1], path[-1])
        # (must convert numpy arrays to tuples to make them hashable)
        original_intermediate_qs = {tuple(q) for q in path[1:-1]}
        for i in range(1, len(shortcut_path) - 1):
            intermediate_q = shortcut_path[i]
            self.assertIn(tuple(intermediate_q), original_intermediate_qs)

    def test_fill_path(self):
        path = [
            np.array([-0.1, 0.0]),
            np.array([0.4, 0.0]),
            np.array([0.6, 0.0]),
        ]

        # perform a fill that adds one intermediate waypoint
        filled_path = utils.fill_path(path, max_dist_between_points=0.3)
        self.assertEqual(len(filled_path), 4)
        np.testing.assert_equal(filled_path[0], path[0])
        np.testing.assert_allclose(
            filled_path[1], np.array([0.2, 0.0]), rtol=0, atol=1e-8
        )
        np.testing.assert_equal(filled_path[2], path[1])
        np.testing.assert_equal(filled_path[3], path[2])

        # perform a fill that adds no intermediate waypoints
        # (dist > adjacent waypoint distance)
        filled_path = utils.fill_path(path, max_dist_between_points=0.75)
        self.assertEqual(len(filled_path), len(path))
        np.testing.assert_equal(filled_path[0], path[0])
        np.testing.assert_equal(filled_path[1], path[1])
        np.testing.assert_equal(filled_path[2], path[2])

        # perform a fill that adds multiple intermediate waypoints
        filled_path = utils.fill_path(path, max_dist_between_points=0.1)
        self.assertEqual(len(filled_path), 8)
        np.testing.assert_equal(filled_path[0], path[0])
        np.testing.assert_allclose(
            filled_path[1], np.array([0.0, 0.0]), rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            filled_path[2], np.array([0.1, 0.0]), rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            filled_path[3], np.array([0.2, 0.0]), rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            filled_path[4], np.array([0.3, 0.0]), rtol=0, atol=1e-8
        )
        np.testing.assert_equal(filled_path[5], path[1])
        np.testing.assert_allclose(
            filled_path[6], np.array([0.5, 0.0]), rtol=0, atol=1e-8
        )
        np.testing.assert_equal(filled_path[7], path[2])


if __name__ == "__main__":
    unittest.main()

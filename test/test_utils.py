import unittest
from pathlib import Path

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl
import mjpl.types

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_BALL_XY_PLANE_XML = _MODEL_DIR / "two_dof_ball.xml"
_JOINTS_XML = _MODEL_DIR / "joints.xml"


def directly_connectable_waypoints() -> list[np.ndarray]:
    """
    Waypoints that can be directly connected between the start and end.
    "X" represents an obstacle.

                  p2 -> p3 --> p4
                  ^            |
                  |     X      |
            p0 -> p1    X      |
                        X      |
                               v
                        p6 <-- p5
    """
    p0 = np.array([0.0, 0.0])
    p1 = np.array([0.25, 0.0])
    p2 = np.array([0.25, 0.75])
    p3 = np.array([0.5, 0.75])
    p4 = np.array([1.0, 0.75])
    p5 = np.array([1.0, -0.75])
    p6 = np.array([0.5, -0.75])
    return [p0, p1, p2, p3, p4, p5, p6]


def shortcuttable_waypoints() -> list[np.ndarray]:
    """
    Waypoints that can benefit from shortcutting.
    "X" represents an obstacle.

                  p2 -> p3 --> p4
                  ^            |
                  |     X      v
            p0 -> p1    X      p5 -> p6
                        X
    """
    p0 = np.array([0.0, 0.0])
    p1 = np.array([0.25, 0.0])
    p2 = np.array([0.25, 0.75])
    p3 = np.array([0.5, 0.75])
    p4 = np.array([1.0, 0.75])
    p5 = np.array([1.0, 0.0])
    p6 = np.array([1.0, 0.25])
    return [p0, p1, p2, p3, p4, p5, p6]


class TestUtils(unittest.TestCase):
    def test_step(self):
        start = np.array([0.0, 0.0])
        target = np.array([0.5, 0.0])

        q_next = mjpl.utils.step(start, target, max_step_dist=5.0)
        np.testing.assert_equal(q_next, target)

        q_next = mjpl.utils.step(start, target, max_step_dist=0.1)
        np.testing.assert_allclose(q_next, np.array([0.1, 0.0]), rtol=0, atol=1e-8)

        with self.assertRaisesRegex(ValueError, "`max_step_dist` must be > 0.0"):
            mjpl.utils.step(start, target, max_step_dist=0.0)
            mjpl.utils.step(start, target, max_step_dist=-1.0)

    def test_site_pose(self):
        model = load_robot_description("ur5e_mj_description")
        data = mujoco.MjData(model)

        mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
        mujoco.mj_kinematics(model, data)

        site_name = "attachment_site"
        pose = mjpl.site_pose(data, site_name)

        site = data.site(site_name)
        np.testing.assert_allclose(site.xpos, pose.translation(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            site.xmat.reshape(3, 3), pose.rotation().as_matrix(), rtol=0, atol=1e-12
        )

    def test_connect_waypoints(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XY_PLANE_XML.as_posix())
        q_idx = list(range(model.nq))
        cr = mjpl.CollisionRuleset(model)

        # Waypoint connection should fail if an obstacle is present.
        waypoints = shortcuttable_waypoints()
        connected_waypoints = mjpl.utils._connect_waypoints(
            model,
            mujoco.MjData(model),
            waypoints,
            q_idx,
            start_idx=1,
            end_idx=5,
            validation_dist=0.1,
            cr=cr,
        )
        self.assertListEqual(waypoints, connected_waypoints)

        waypoints = [
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([2.0, 1.0]),
            # This is outside of the model's joint limits
            np.array([3.0, 1.0]),
        ]

        # Test a valid waypoint connection.
        connected_waypoints = mjpl.utils._connect_waypoints(
            model,
            mujoco.MjData(model),
            waypoints,
            q_idx,
            start_idx=0,
            end_idx=2,
            validation_dist=0.25,
            cr=cr,
        )
        self.assertListEqual(
            connected_waypoints, [waypoints[0], waypoints[2], waypoints[3]]
        )

        # Test an invalid waypoint connection. `end_idx` corresponds to a waypoint
        # that violates joint limits, so this call should do nothing (i.e., the
        # returned waypoints should be an unmodified copy of the original).
        connected_waypoints = mjpl.utils._connect_waypoints(
            model,
            mujoco.MjData(model),
            waypoints,
            q_idx,
            start_idx=1,
            end_idx=3,
            validation_dist=0.25,
            cr=cr,
        )
        self.assertListEqual(connected_waypoints, waypoints)

        data = mujoco.MjData(model)
        with self.assertRaisesRegex(ValueError, "`validation_dist` must be > 0"):
            mjpl.utils._connect_waypoints(
                model,
                data,
                waypoints,
                q_idx,
                start_idx=0,
                end_idx=2,
                validation_dist=0.0,
                cr=cr,
            )
            mjpl.utils._connect_waypoints(
                model,
                data,
                waypoints,
                q_idx,
                start_idx=0,
                end_idx=2,
                validation_dist=-1.0,
                cr=cr,
            )

    def test_shortcut(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XY_PLANE_XML.as_posix())
        planning_joints = ["ball_slide_x", "ball_slide_y"]
        cr = mjpl.CollisionRuleset(model)

        waypoints = directly_connectable_waypoints()
        path = mjpl.types.Path(
            q_init=waypoints[0], waypoints=waypoints, joints=planning_joints
        )

        # The first and last waypoints in the path can be connected directly.
        shortened_path = mjpl.shortcut(model, path, cr, seed=5)
        np.testing.assert_equal(shortened_path.q_init, path.q_init)
        self.assertListEqual(
            shortened_path.waypoints, [path.waypoints[0], path.waypoints[-1]]
        )
        self.assertListEqual(shortened_path.joints, path.joints)

        # Add a final waypoint to the path that violates joint limits.
        # This means that after enough tries, the first and penultimate
        # (i.e., last valid) waypoints can be directly connected
        invalid_waypoints = waypoints + [np.array([3.0, -5.0])]
        invalid_path = mjpl.types.Path(
            q_init=waypoints[0], waypoints=invalid_waypoints, joints=planning_joints
        )
        shortened_path = mjpl.shortcut(
            model, invalid_path, cr, max_attempts=20, seed=42
        )
        np.testing.assert_equal(shortened_path.q_init, path.q_init)
        self.assertListEqual(
            shortened_path.waypoints,
            [
                invalid_path.waypoints[0],
                invalid_path.waypoints[-2],
                invalid_path.waypoints[-1],
            ],
        )
        self.assertListEqual(shortened_path.joints, invalid_path.joints)

        # Test shortcutting on a path that can be shortened, but is not
        # directly connectable from the start to end.
        waypoints = shortcuttable_waypoints()
        path = mjpl.types.Path(
            q_init=waypoints[0], waypoints=waypoints, joints=planning_joints
        )
        shortened_path = mjpl.shortcut(model, path, cr, seed=5)
        np.testing.assert_equal(shortened_path.q_init, path.q_init)
        self.assertListEqual(
            shortened_path.waypoints,
            [
                path.waypoints[0],
                path.waypoints[2],
                path.waypoints[4],
                path.waypoints[6],
            ],
        )
        self.assertListEqual(shortened_path.joints, path.joints)

    def test_shortcut_6dof(self):
        model = load_robot_description("ur5e_mj_description")
        cr = mjpl.CollisionRuleset(model)
        seed = 42

        # All joints can be represented explicitly or as an empty list.
        arm_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        for joints in ([], arm_joints):
            # Make a path starting from the home config that connects to various random valid configs.
            waypoints = [model.keyframe("home").qpos.copy()]
            unique_waypoints = {tuple(waypoints[0])}
            for i in range(5):
                q_rand = mjpl.random_valid_config(
                    model, np.zeros(model.nq), seed + i, joints, cr
                )
                waypoints.append(q_rand)
                hashable_q_rand = tuple(q_rand)
                self.assertNotIn(hashable_q_rand, unique_waypoints)
                unique_waypoints.add(tuple(q_rand))
            path = mjpl.types.Path(
                q_init=waypoints[0], waypoints=waypoints, joints=joints
            )

            # Perform shortcutting. The path should now be shorter, but still start
            # and end at the same waypoints.
            shortcut_path = mjpl.shortcut(model, path, cr, seed=seed)
            np.testing.assert_equal(shortcut_path.q_init, path.q_init)
            self.assertLess(len(shortcut_path.waypoints), len(path.waypoints))
            self.assertGreaterEqual(len(shortcut_path.waypoints), 2)
            np.testing.assert_equal(shortcut_path.waypoints[0], path.waypoints[0])
            np.testing.assert_equal(shortcut_path.waypoints[-1], path.waypoints[-1])
            # (must convert numpy arrays to tuples to make them hashable)
            original_intermediate_qs = {tuple(q) for q in path.waypoints[1:-1]}
            for intermediate_q in shortcut_path.waypoints[1:-1]:
                self.assertIn(tuple(intermediate_q), original_intermediate_qs)
            self.assertListEqual(shortcut_path.joints, path.joints)

    def test_qpos_idx(self):
        model = mujoco.MjModel.from_xml_path(_JOINTS_XML.as_posix())

        # Querying all joints in the model should correspond to the full mujoco.MjData.qpos
        indices = mjpl.qpos_idx(
            model, ["slide_joint", "free_joint", "hinge_joint", "ball_joint"]
        )
        self.assertListEqual(indices, list(range(model.nq)))

        indices = mjpl.qpos_idx(model, ["slide_joint"])
        self.assertListEqual(indices, [0])

        indices = mjpl.qpos_idx(model, ["free_joint"])
        self.assertListEqual(indices, [1, 2, 3, 4, 5, 6, 7])

        indices = mjpl.qpos_idx(model, ["hinge_joint"])
        self.assertListEqual(indices, [8])

        indices = mjpl.qpos_idx(model, ["ball_joint"])
        self.assertListEqual(indices, [9, 10, 11, 12])

        # Make sure index order matches order of joints in the query.
        indices = mjpl.qpos_idx(model, ["ball_joint", "hinge_joint", "free_joint"])
        self.assertListEqual(indices, [9, 10, 11, 12, 8, 1, 2, 3, 4, 5, 6, 7])

        self.assertListEqual(mjpl.qpos_idx(model, [], default_to_full=False), [])
        self.assertListEqual(
            mjpl.qpos_idx(model, [], default_to_full=True), list(range(model.nq))
        )

    def test_is_valid_config(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XY_PLANE_XML.as_posix())
        data = mujoco.MjData(model)
        cr = mjpl.CollisionRuleset(model)

        # Valid configuration (within joint limits, obeys CollisionRuleset)
        data.qpos = np.zeros(model.nq)
        self.assertTrue(mjpl.utils.is_valid_config(model, data, cr))

        # Invalid configuration (violates joint limits)
        data.qpos = np.array([2.5, 0.0])
        self.assertFalse(mjpl.utils.is_valid_config(model, data, cr))

        # Invalid configuration (within joint limits, violates CollisionRuleset)
        data.qpos = np.array([0.6, 0.0])
        self.assertFalse(mjpl.utils.is_valid_config(model, data, cr))

        # Disabling the CollisionRuleset in the validity check makes a configuration
        # that is within joint limits but in collision a valid configuration.
        data.qpos = np.array([0.6, 0.0])
        self.assertTrue(mjpl.utils.is_valid_config(model, data, None))

    def test_random_valid_config(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XY_PLANE_XML.as_posix())
        data = mujoco.MjData(model)
        cr = mjpl.CollisionRuleset(model)

        seed = 42

        # Using the same seed should give the same result across multiple calls if all
        # other args are consistent.
        q_init = np.zeros(model.nq)
        q_rand_first = mjpl.random_valid_config(model, q_init, seed, [], cr)
        q_rand_second = mjpl.random_valid_config(model, q_init, seed, [], cr)
        np.testing.assert_equal(q_rand_first, q_rand_second)
        data.qpos = q_rand_first
        self.assertTrue(mjpl.utils.is_valid_config(model, data, cr))

        # Specifying a subset of joints means some values in q_init shouldn't be modified.
        q_init = np.zeros(model.nq)
        modifiable_joints = ["ball_slide_y"]
        q_rand = mjpl.random_valid_config(model, q_init, seed, modifiable_joints, cr)
        unchanged_idx = mjpl.qpos_idx(model, ["ball_slide_x"])
        np.testing.assert_equal(q_rand[unchanged_idx], q_init[unchanged_idx])
        data.qpos = q_rand
        self.assertTrue(mjpl.utils.is_valid_config(model, data, cr))


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_BALL_XY_PLANE_XML = _MODEL_DIR / "two_dof_ball.xml"


def directly_connectable_path() -> list[np.ndarray]:
    """
    Path that can be directly connected between the start and end.
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


def shortcuttable_path() -> list[np.ndarray]:
    """
    Path that can benefit from shortcutting.
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
        planning_joints = [
            model.joint("ball_slide_x").id,
            model.joint("ball_slide_y").id,
        ]
        jg = mjpl.JointGroup(model, planning_joints)
        cr = mjpl.CollisionRuleset(model)

        # Waypoint connection should fail if an obstacle is present.
        path = shortcuttable_path()
        connected_path = mjpl.utils._connect_waypoints(
            path,
            start_idx=1,
            end_idx=5,
            validation_dist=0.1,
            jg=jg,
            data=mujoco.MjData(model),
            cr=cr,
        )
        self.assertListEqual(path, connected_path)

        path = [
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([2.0, 1.0]),
            # This is outside of the model's joint limits
            np.array([3.0, 1.0]),
        ]

        # Connect waypoints with validation checking.
        connected_path = mjpl.utils._connect_waypoints(
            path,
            start_idx=0,
            end_idx=2,
            validation_dist=0.25,
            jg=jg,
            data=mujoco.MjData(model),
            cr=cr,
        )
        self.assertListEqual(connected_path, [path[0], path[2], path[3]])

        # Connect with validation checking. Since `end_idx` corresponds to
        # a waypoint that violates the joint limits, this should fail
        # (i.e., the returned path is an unmodified copy of the original).
        connected_path = mjpl.utils._connect_waypoints(
            path,
            start_idx=1,
            end_idx=3,
            validation_dist=0.25,
            jg=jg,
            data=mujoco.MjData(model),
            cr=cr,
        )
        self.assertListEqual(connected_path, path)

        # Connecting without validation checking on the waypoint that violates
        # joint limits should work.
        connected_path = mjpl.utils._connect_waypoints(
            path,
            start_idx=1,
            end_idx=3,
            validation_dist=0.1,
            jg=None,
            data=None,
        )
        self.assertListEqual(connected_path, [path[0], path[1], path[3]])

        with self.assertRaisesRegex(
            ValueError, "`jg` and `data` must either be None or not None"
        ):
            mjpl.utils._connect_waypoints(
                path, start_idx=0, end_idx=2, jg=None, data=mujoco.MjData(model)
            )
            mjpl.utils._connect_waypoints(
                path, start_idx=0, end_idx=2, jg=jg, data=None
            )

        with self.assertRaisesRegex(ValueError, "`validation_dist` must be > 0"):
            mjpl.utils._connect_waypoints(
                path, start_idx=0, end_idx=2, validation_dist=0.0
            )
            mjpl.utils._connect_waypoints(
                path, start_idx=0, end_idx=2, validation_dist=-1.0
            )

    def test_shortcut(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XY_PLANE_XML.as_posix())
        planning_joints = [
            model.joint("ball_slide_x").id,
            model.joint("ball_slide_y").id,
        ]
        jg = mjpl.JointGroup(model, planning_joints)
        cr = mjpl.CollisionRuleset(model)

        path = directly_connectable_path()

        # The first and last waypoints in the path can be connected directly.
        shortened_path = mjpl.shortcut(path, jg, cr, seed=5)
        self.assertListEqual(shortened_path, [path[0], path[-1]])

        # Add a final point to the path that violates joint limits.
        # This means that after enough tries, the first and penultimate
        # (i.e., last valid) waypoints can be directly connected
        invalid_path = path + [np.array([3.0, -5.0])]
        shortened_path = mjpl.shortcut(invalid_path, jg, cr, max_attempts=20, seed=42)
        self.assertListEqual(
            shortened_path, [invalid_path[0], invalid_path[-2], invalid_path[-1]]
        )

        # Test shortcutting on a path that can be shortened, but is not
        # directly connectable from the start to end.
        path = shortcuttable_path()
        shortened_path = mjpl.shortcut(path, jg, cr, seed=5)
        self.assertListEqual(shortened_path, [path[0], path[2], path[4], path[6]])

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
        jg = mjpl.JointGroup(model, arm_joint_ids)
        cr = mjpl.CollisionRuleset(model)

        seed = 42

        # Make a path starting from the home config that connects to various random valid configs.
        path = [model.keyframe("home").qpos.copy()]
        rng = np.random.default_rng(seed=seed)
        random_waypoints = [
            mjpl.random_valid_config(rng, jg, data, cr) for _ in range(5)
        ]
        path.extend(random_waypoints)

        # Perform shortcutting. The path should now be shorter, but still start
        # and end at the same waypoints.
        shortcut_path = mjpl.shortcut(path, jg, cr, seed=seed)
        self.assertLess(len(shortcut_path), len(path))
        self.assertGreaterEqual(len(shortcut_path), 2)
        np.testing.assert_equal(shortcut_path[0], path[0])
        np.testing.assert_equal(shortcut_path[-1], path[-1])
        # (must convert numpy arrays to tuples to make them hashable)
        original_intermediate_qs = {tuple(q) for q in path[1:-1]}
        for intermediate_q in shortcut_path[1:-1]:
            self.assertIn(tuple(intermediate_q), original_intermediate_qs)


if __name__ == "__main__":
    unittest.main()

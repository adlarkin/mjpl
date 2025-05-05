import unittest
from pathlib import Path

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl

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

    def test_all_joints(self):
        model = load_robot_description("ur5e_mj_description")
        expected_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.assertListEqual(mjpl.all_joints(model), expected_joints)

        model = model.from_xml_path(_JOINTS_XML.as_posix())
        expected_joints = [
            "slide_joint",
            "free_joint",
            "hinge_joint",
            "ball_joint",
        ]
        self.assertListEqual(mjpl.all_joints(model), expected_joints)

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
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]

        # Waypoint connection should fail if it violates the collision constraint.
        waypoints = shortcuttable_waypoints()
        connected_waypoints = mjpl.utils._connect_waypoints(
            waypoints,
            start_idx=1,
            end_idx=5,
            validation_dist=0.1,
            constraints=constraints,
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
            waypoints,
            start_idx=0,
            end_idx=2,
            validation_dist=0.25,
            constraints=constraints,
        )
        self.assertListEqual(
            connected_waypoints, [waypoints[0], waypoints[2], waypoints[3]]
        )

        # Waypoint connection should fail if it violates the joint limit constraint.
        connected_waypoints = mjpl.utils._connect_waypoints(
            waypoints,
            start_idx=1,
            end_idx=3,
            validation_dist=0.25,
            constraints=constraints,
        )
        self.assertListEqual(connected_waypoints, waypoints)

        with self.assertRaisesRegex(ValueError, "`validation_dist` must be > 0"):
            mjpl.utils._connect_waypoints(
                waypoints,
                start_idx=0,
                end_idx=2,
                validation_dist=0.0,
                constraints=constraints,
            )
            mjpl.utils._connect_waypoints(
                waypoints,
                start_idx=0,
                end_idx=2,
                validation_dist=-1.0,
                constraints=constraints,
            )

    def test_shortcut(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XY_PLANE_XML.as_posix())
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]

        waypoints = directly_connectable_waypoints()

        # The first and last waypoints can be connected directly without violating constraints.
        shortened_waypoints = mjpl.shortcut(waypoints, constraints, seed=5)
        self.assertListEqual(shortened_waypoints, [waypoints[0], waypoints[-1]])

        # Add a final waypoint that violates the joint limit constraint. This means that
        # after enough tries, the first and penultimate (i.e., last valid) waypoints can
        # be directly connected.
        invalid_waypoints = waypoints + [np.array([3.0, -5.0])]
        shortened_waypoints = mjpl.shortcut(
            invalid_waypoints, constraints, max_attempts=20, seed=42
        )
        self.assertListEqual(
            shortened_waypoints,
            [invalid_waypoints[0], invalid_waypoints[-2], invalid_waypoints[-1]],
        )

        # Test shortcutting on waypoints that can be shortened, but are not
        # directly connectable from the start to end.
        waypoints = shortcuttable_waypoints()
        shortened_waypoints = mjpl.shortcut(waypoints, constraints, seed=5)
        self.assertListEqual(
            shortened_waypoints,
            [
                waypoints[0],
                waypoints[2],
                waypoints[4],
                waypoints[6],
            ],
        )

    def test_shortcut_6dof(self):
        model = load_robot_description("ur5e_mj_description")
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]
        seed = 42

        # Make a path starting from the home config that connects to various random valid configs.
        waypoints = [model.keyframe("home").qpos.copy()]
        unique_waypoints = {tuple(waypoints[0])}
        for i in range(5):
            q_rand = mjpl.random_config(
                model, np.zeros(model.nq), mjpl.all_joints(model), seed + i, constraints
            )
            waypoints.append(q_rand)
            hashable_q_rand = tuple(q_rand)
            self.assertNotIn(hashable_q_rand, unique_waypoints)
            unique_waypoints.add(tuple(q_rand))

        # Perform shortcutting. There should be fewer waypoints in the list, but the
        # start and end waypoints should be the same.
        shortcut_waypoints = mjpl.shortcut(waypoints, constraints, seed=seed)
        self.assertLess(len(shortcut_waypoints), len(waypoints))
        self.assertGreaterEqual(len(shortcut_waypoints), 2)
        np.testing.assert_equal(shortcut_waypoints[0], waypoints[0])
        np.testing.assert_equal(shortcut_waypoints[-1], waypoints[-1])
        # (must convert numpy arrays to tuples to make them hashable)
        original_intermediate_qs = {tuple(q) for q in waypoints[1:-1]}
        for intermediate_q in shortcut_waypoints[1:-1]:
            self.assertIn(tuple(intermediate_q), original_intermediate_qs)

        # The shortcutted waypoints should still obey all constraints.
        for wp in shortcut_waypoints:
            self.assertTrue(mjpl.obeys_constraints(wp, constraints))

    def test_qpos_idx(self):
        model = mujoco.MjModel.from_xml_path(_JOINTS_XML.as_posix())

        # Querying all joints in the model should correspond to the full mujoco.MjData.qpos
        indices = mjpl.qpos_idx(model, mjpl.all_joints(model))
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

        self.assertListEqual(mjpl.qpos_idx(model, []), [])

    def test_qvel_idx(self):
        model = mujoco.MjModel.from_xml_path(_JOINTS_XML.as_posix())

        # Querying all joints in the model should correspond to the full mujoco.MjData.qvel
        indices = mjpl.qvel_idx(model, mjpl.all_joints(model))
        self.assertListEqual(indices, list(range(model.nv)))

        indices = mjpl.qvel_idx(model, ["slide_joint"])
        self.assertListEqual(indices, [0])

        indices = mjpl.qvel_idx(model, ["free_joint"])
        self.assertListEqual(indices, [1, 2, 3, 4, 5, 6])

        indices = mjpl.qvel_idx(model, ["hinge_joint"])
        self.assertListEqual(indices, [7])

        indices = mjpl.qvel_idx(model, ["ball_joint"])
        self.assertListEqual(indices, [8, 9, 10])

        # Make sure index order matches order of joints in the query.
        indices = mjpl.qvel_idx(model, ["ball_joint", "hinge_joint", "free_joint"])
        self.assertListEqual(indices, [8, 9, 10, 7, 1, 2, 3, 4, 5, 6])

        self.assertListEqual(mjpl.qvel_idx(model, []), [])

    def test_random_config(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XY_PLANE_XML.as_posix())
        constraints = [
            mjpl.JointLimitConstraint(model),
            mjpl.CollisionConstraint(model),
        ]

        seed = 42

        # Using the same seed should give the same result across multiple calls if all
        # other args are consistent.
        joints = mjpl.all_joints(model)
        q_init = np.zeros(model.nq)
        q_rand_first = mjpl.random_config(model, q_init, joints, seed, constraints)
        q_rand_second = mjpl.random_config(model, q_init, joints, seed, constraints)
        np.testing.assert_equal(q_rand_first, q_rand_second)
        self.assertTrue(mjpl.obeys_constraints(q_rand_first, constraints))
        self.assertTrue(mjpl.obeys_constraints(q_rand_second, constraints))

        # Specifying a subset of joints means some values in q_init shouldn't be modified.
        q_init = np.zeros(model.nq)
        modifiable_joints = ["ball_slide_y"]
        q_rand = mjpl.random_config(model, q_init, modifiable_joints, seed, constraints)
        unchanged_idx = mjpl.qpos_idx(model, ["ball_slide_x"])
        np.testing.assert_equal(q_rand[unchanged_idx], q_init[unchanged_idx])
        self.assertTrue(mjpl.obeys_constraints(q_rand, constraints))


if __name__ == "__main__":
    unittest.main()

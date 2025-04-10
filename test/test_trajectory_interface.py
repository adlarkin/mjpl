import unittest

import mujoco
import numpy as np
from scipy.interpolate import make_interp_spline

import mjpl
from mjpl.trajectory.trajectory_interface import (
    Trajectory,
    TrajectoryGenerator,
    _closest_segment,
)


class MockTrajectoryGenerator(TrajectoryGenerator):
    """Mock TrajectoryGenerator class."""

    path = [
        np.zeros(3),
        np.zeros(3) + 2,
    ]
    positions = [
        np.zeros(3),
        np.zeros(3) + 1,
        np.zeros(3) + 2,
    ]
    velocities = [
        np.zeros(3),
        np.zeros(3) + 1,
        np.zeros(3) + 2,
    ]
    accelerations = [
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
    ]

    def _build(self, path: list[np.ndarray]) -> Trajectory:
        return Trajectory(
            dt=0.002,
            positions=self.positions,
            velocities=self.velocities,
            accelerations=self.accelerations,
        )


class TestTrajectoryInterface(unittest.TestCase):
    def test_generate_trajectory(self):
        generator = MockTrajectoryGenerator()

        trajectory = generator.generate_trajectory(MockTrajectoryGenerator.path)
        self.assertTrue(
            len(trajectory.positions), len(MockTrajectoryGenerator.positions)
        )
        for i in range(len(trajectory.positions)):
            np.testing.assert_equal(
                trajectory.positions[i], MockTrajectoryGenerator.positions[i]
            )
        self.assertTrue(
            len(trajectory.velocities), len(MockTrajectoryGenerator.velocities)
        )
        for i in range(len(trajectory.velocities)):
            np.testing.assert_equal(
                trajectory.velocities[i], MockTrajectoryGenerator.velocities[i]
            )
        self.assertTrue(
            len(trajectory.accelerations), len(MockTrajectoryGenerator.accelerations)
        )
        for i in range(len(trajectory.accelerations)):
            np.testing.assert_equal(
                trajectory.accelerations[i], MockTrajectoryGenerator.accelerations[i]
            )

        xml_str = "<mujoco></mujoco>"
        model = mujoco.MjModel.from_xml_string(xml_str)
        with self.assertRaisesRegex(ValueError, "must all be defined or `None`"):
            generator.generate_trajectory(
                MockTrajectoryGenerator.path, q_init=np.zeros(3)
            )
            generator.generate_trajectory(
                MockTrajectoryGenerator.path, cr=mjpl.CollisionRuleset(model)
            )
            generator.generate_trajectory(
                MockTrajectoryGenerator.path, jg=mjpl.JointGroup(model, joint_ids=[])
            )

    def test_closest_segment(self):
        path = [
            np.array([0, 0]),
            np.array([1, 1]),
            np.array([2, 1]),
            np.array([2, 0]),
        ]
        splx = [0, 1 / 3, 2 / 3, 1]
        spline = make_interp_spline(splx, path)

        self.assertEqual(_closest_segment(spline(1 / 6), path), (0, 1))
        self.assertEqual(_closest_segment(spline(0.5), path), (1, 2))
        self.assertEqual(_closest_segment(spline(2.5 / 3), path), (2, 3))


if __name__ == "__main__":
    unittest.main()

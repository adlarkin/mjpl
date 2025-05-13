import unittest
from pathlib import Path

import mujoco
import numpy as np

import mjpl

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_TWO_DOF_BALL_XML = _MODEL_DIR / "two_dof_ball.xml"


class TestJointLimitConstraint(unittest.TestCase):
    def test_constraint(self):
        model = mujoco.MjModel.from_xml_path(_TWO_DOF_BALL_XML.as_posix())

        # Constraint that uses joint limits from the MuJoCo model.
        constraint = mjpl.JointLimitConstraint(model)

        # Test a configuration that does not violate the constraint.
        q = np.array([0.0, 0.0])
        self.assertTrue(constraint.valid_config(q))
        q_constrained = constraint.apply(np.array([]), q)
        self.assertIsNotNone(q_constrained)
        np.testing.assert_equal(q_constrained, q)

        # Test a configuration that violates the constraint.
        q = np.array([2.5, 0.0])
        self.assertFalse(constraint.valid_config(q))
        self.assertIsNone(constraint.apply(np.array([]), q))


if __name__ == "__main__":
    unittest.main()

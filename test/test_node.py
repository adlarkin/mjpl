import numpy as np
import unittest

from mj_maniPlan.rrt import Node


class TestNode(unittest.TestCase):
    def test_eq(self):
        a = Node(np.array([0,1,2]), None)
        b = Node(np.array([0,1,2]), a)
        c = Node(np.array([3,4,5]), a)
        d = Node(np.array([3,4,5]), b)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertNotEqual(b, c)
        self.assertEqual(c, d)


if __name__ == '__main__':
    unittest.main()

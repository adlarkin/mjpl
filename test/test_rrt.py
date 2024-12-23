import mujoco
import numpy as np
import unittest
from pathlib import Path

from mj_maniPlan.sampling import HaltonSampler
import mj_maniPlan.rrt as rrt


_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "models"
_BALL_XML = _MODEL_DIR / "ball.xml"
_BALL_XY_PLANE_XML = _MODEL_DIR / "ball_xy_plane.xml"


class TestRRT(unittest.TestCase):
    def load_ball_with_obstacle_model(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XML.as_posix())

        self.obstacle = model.geom('wall_obstacle')

        joint_names = [ "ball_slide_x" ]
        options = rrt.RRTOptions(
            joint_names=joint_names,
            max_planning_time=5.0,
            epsilon=0.1,
            shortcut_filler_epsilon=0.1,
            rng=HaltonSampler(len(joint_names), seed=42)
        )

        # Set the initial joint configuration.
        # We're testing with a simple model: a ball that can slide along the x-axis.
        # So there's only one value in data.qpos (the ball's x position)
        self.q_init = np.array([-0.1])
        data = mujoco.MjData(model)
        data.qpos[0] = self.q_init[0]
        mujoco.mj_kinematics(model, data)

        self.planner = rrt.RRT(options, model, data)

    def load_ball_sliding_along_xy_model(self):
        model = mujoco.MjModel.from_xml_path(_BALL_XY_PLANE_XML.as_posix())

        joint_names = [
            "ball_slide_x",
            "ball_slide_y",
        ]
        options = rrt.RRTOptions(
            joint_names=joint_names,
            max_planning_time=5.0,
            epsilon=0.2,
            # TODO: make this a different value then epsilon to capture it in tests?
            shortcut_filler_epsilon=0.2,
            rng=HaltonSampler(len(joint_names), seed=42)
        )

        # Set the initial joint configuration.
        self.q_init = np.array([-0.1, 0.0])
        data = mujoco.MjData(model)
        data.qpos = self.q_init.copy()
        mujoco.mj_kinematics(model, data)

        self.planner = rrt.RRT(options, model, data)

    def test_extend(self):
        self.load_ball_with_obstacle_model()

        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        q_goal = np.array([0.5])
        expected_q_extended = np.array([0.0])
        extended_node = self.planner.extend(q_goal, tree)
        self.assertIsNotNone(extended_node)
        self.assertEqual(len(tree.nodes), 2)
        self.assertAlmostEqual(extended_node.q, expected_q_extended, places=9)

        # Check the path starting from the extended node.
        # This implicitly checks the extended node's parent.
        expected_path = [
            expected_q_extended,
            self.q_init,
        ]
        tree.set_path_root(extended_node)
        path = tree.get_path()
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            self.assertAlmostEqual(path[i][0], expected_path[i][0], places=9)

        # extending towards a q that is already in the tree should do nothing
        existing_nodes = tree.nodes.copy()
        same_extended_node = self.planner.extend(extended_node.q, tree)
        self.assertIsNotNone(same_extended_node)
        self.assertSetEqual(tree.nodes, existing_nodes)
        # The assertSetEqual above runs the equality operator on all elements of the set.
        # Since node equality is defined as having the same q, we should take the check a
        # step further to ensure that same_extended_node and extended_node are the same object.
        self.assertIs(same_extended_node, extended_node)

        # Test extend where the "nearest node" is already given
        tree = rrt.Tree()
        root_node = rrt.Node(np.array([0.0]), None)
        tree.add_node(root_node)
        tree.add_node(rrt.Node(np.array([2.0]), root_node))
        extended_node = self.planner.extend(np.array([3.0]), tree, root_node)
        self.assertIsNotNone(extended_node)
        self.assertIs(extended_node.parent, root_node)
        self.assertAlmostEqual(extended_node.q[0], 0.1, places=9)

    def test_connect(self):
        self.load_ball_with_obstacle_model()

        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        q_goal = np.array([0.15])
        goal_node = self.planner.connect(q_goal, tree)

        expected_qs_in_tree = [
            self.q_init,
            np.array([0.0]),
            np.array([0.1]),
            q_goal,
        ]
        for q_tree in expected_qs_in_tree:
            n = tree.nearest_neighbor(q_tree)
            self.assertAlmostEqual(n.q[0], q_tree[0], places=9)

        # Check the path from the last connected node.
        # This implicitly checks each connected node's parent.
        expected_path = expected_qs_in_tree[::-1]
        tree.set_path_root(goal_node)
        path = tree.get_path()
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            self.assertAlmostEqual(path[i][0], expected_path[i][0], places=9)

        # connecting towards a q that is already in the tree should do nothing
        existing_nodes = tree.nodes.copy()
        same_connected_node = self.planner.connect(goal_node.q, tree)
        self.assertSetEqual(tree.nodes, existing_nodes)
        # The assertSetEqual above runs the equality operator on all elements of the set.
        # Since node equality is defined as having the same q, we should take the check a
        # step further to ensure that same_extended_node and extended_node are the same object.
        self.assertIs(same_connected_node, goal_node)

    def test_extend_and_connect_into_obstacle(self):
        self.load_ball_with_obstacle_model()

        tree = rrt.Tree()
        tree.add_node(rrt.Node(self.q_init, None))

        # Try to connect to the maximum joint value for the ball.
        # There is an obstacle in the way, so the final connection should be just before
        # where the obstacle lies.
        q_goal = np.array([1.0])
        connected_node = self.planner.connect(q_goal, tree)
        obstacle_min_x = self.obstacle.pos[0] - self.obstacle.size[0]
        self.assertLess(connected_node.q[0], obstacle_min_x)

        # Try to extend towards the goal.
        # This should not work since the extension is in collision.
        existing_nodes = tree.nodes.copy()
        extended_node = self.planner.extend(q_goal, tree)
        self.assertIsNone(extended_node)
        self.assertSetEqual(tree.nodes, existing_nodes)

    def test_get_path(self):
        self.load_ball_with_obstacle_model()

        q_new = np.array([0.15])

        start_tree = rrt.Tree()
        start_tree.add_node(rrt.Node(self.q_init, None))
        connected_node_a = self.planner.connect(q_new, start_tree)
        start_tree.set_path_root(connected_node_a)

        goal_tree = rrt.Tree()
        goal_tree.add_node(rrt.Node(np.array([0.5]), None))
        connected_node_b = self.planner.connect(q_new, goal_tree)
        goal_tree.set_path_root(connected_node_b)

        path = self.planner.get_path(start_tree, goal_tree)
        expected_path = [
            self.q_init,
            np.array([0.0]),
            np.array([0.1]),
            np.array([0.15]),
            np.array([0.2]),
            np.array([0.3]),
            np.array([0.4]),
            np.array([0.5]),
        ]
        self.assertEqual(len(path), len(expected_path))
        for i in range(len(path)):
            self.assertAlmostEqual(path[i][0], expected_path[i][0], places=9)

    def test_run_rrt(self):
        self.load_ball_with_obstacle_model()

        q_goal = np.array([0.35])
        path = self.planner.plan(q_goal)
        self.assertIsNotNone(path)

        # The path should start at q_init and end at q_goal
        self.assertTrue(np.array_equal(path[0], self.q_init))
        self.assertTrue(np.array_equal(path[-1], q_goal))

        # The path should be strictly increasing to q_goal
        for i in range(1, len(path)):
            self.assertGreater(path[i][0], path[i-1][0])

    def test_trivial_rrt(self):
        self.load_ball_with_obstacle_model()

        # If we plan to a goal that is directly reachable, the planner should make the direct connection and exit
        q_goal = np.array([-0.05])
        path = self.planner.plan(q_goal)
        self.assertEqual(len(path), 2)
        self.assertTrue(np.array_equal(path[0], self.q_init))
        self.assertTrue(np.array_equal(path[1], q_goal))

    def test_shortcut(self):
        self.load_ball_sliding_along_xy_model()

        '''
        Suboptimal path that can benefit from shortcutting

                          n_3 -> n_4 -> n_5
                           ^             |
                           |             v
            n_0 -> n_1 -> n_2           n_6 -> n_7
        '''
        tree = rrt.Tree()
        n_0 = rrt.Node(self.q_init, None)
        n_1 = rrt.Node(np.array([0.0, 0.0]), n_0)
        n_2 = rrt.Node(np.array([0.1, 0.0]), n_1)
        n_3 = rrt.Node(np.array([0.1, 0.1]), n_2)
        n_4 = rrt.Node(np.array([0.15, 0.1]), n_3)
        n_5 = rrt.Node(np.array([0.2, 0.1]), n_4)
        n_6 = rrt.Node(np.array([0.2, 0.0]), n_5)
        n_7 = rrt.Node(np.array([0.3, 0.0]), n_6)
        nodes = [ n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7 ]
        for n in nodes:
            tree.add_node(n)

        tree.set_path_root(n_7)
        path = tree.get_path()
        path.reverse()
        self.assertEqual(len(path), len(nodes))
        for i in range(len(path)):
            self.assertTrue(np.array_equal(path[i], nodes[i].q))

        original_path = path.copy()

        expected_shortcut_path = [
            n_0.q,
            n_1.q,
            n_2.q,
            n_6.q,
            n_7.q,
        ]
        shortcut_path = self.planner.shortcut(path, start_idx=2, end_idx=6)
        self.assertEqual(len(shortcut_path), len(expected_shortcut_path))
        for i in range(len(shortcut_path)):
            self.assertTrue(np.array_equal(shortcut_path[i], expected_shortcut_path[i]))

        shortcut_path = self.planner.shortcut(path, start_idx=0, end_idx=7)
        self.assertEqual(len(shortcut_path), 3)
        # The first and last points of the shortcut path should match the first
        # and last points of the original path
        self.assertTrue(np.array_equal(shortcut_path[0], path[0]))
        self.assertTrue(np.array_equal(shortcut_path[-1], path[-1]))
        # start_idx can be directly connected to end_idx, so interpolated waypoints are
        # added at a distance of rrt.RRTOptions.epsilon w.r.t. start_idx
        self.assertTrue((abs(shortcut_path[1] - np.array([0.1, 0])) <= 1e-9).all())

        # shortcutting two adjacent waypoints shouldn't modify the original path
        unmodified_path = self.planner.shortcut(path, start_idx=6, end_idx=7)
        self.assertEqual(len(unmodified_path), len(path))
        for i in range(len(unmodified_path)):
            self.assertTrue(np.array_equal(unmodified_path[i], path[i]))

        # make sure invalid kwargs are caught
        with self.assertRaisesRegex(ValueError, 'Invalid kwargs'):
            self.planner.shortcut(path)
            self.planner.shortcut(path, num_attempts=1, start_idx=5, end_idx=6)
            self.planner.shortcut(path, num_attempts=1, start_idx=5)
            self.planner.shortcut(path, num_attempts=1, end_idx=5)
            self.planner.shortcut(path, start_idx=5)
            self.planner.shortcut(path, end_idx=5)

        # shortcutting should create a new path and not modify the original
        self.assertEqual(len(original_path), len(path))
        for i in range(len(original_path)):
            self.assertTrue(np.array_equal(original_path[i], path[i]))


if __name__ == '__main__':
    unittest.main()

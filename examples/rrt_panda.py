'''
Example of how to run RRT on a franka panda and visualize the paths
(with and without shortcutting).
'''

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
from scipy.interpolate import make_interp_spline

from mj_maniPlan.rrt import (
    RRT,
    RRTOptions,
)
import mj_maniPlan.utils as utils
import mj_maniPlan.visualization as viz


_HERE = Path(__file__).parent
_PANDA_XML = _HERE.parent / "models" / "franka_emika_panda" / "scene_with_obstacles.xml"

_EE_SITE = 'ee_site'


# Naive timing generation for a path, which can be used for something like B-spline interpolation.
# Configuration distance between two adjacent path waypoints - q_curr, q_next - is used as a notion
# for the time it takes to move from q_curr to q_next.
def generate_path_timing(path):
    timing = [0.0]
    for i in range(1, len(path)):
        timing.append(timing[-1] + utils.configuration_distance(path[i-1], path[i]))
    # scale to [0..1]
    return np.interp(timing, (timing[0], timing[-1]), (0, 1))

def update_joint_config(q: np.ndarray, qpos_addrs, model: mujoco.MjModel, data: mujoco.MjData):
    data.qpos[qpos_addrs] = q
    mujoco.mj_kinematics(model, data)


if __name__ == '__main__':
    model = mujoco.MjModel.from_xml_path(_PANDA_XML.as_posix())
    data = mujoco.MjData(model)

    # The joints to sample during planning.
    # Since this example executes planning for the arm,
    # the finger joints of the gripper are excluded.
    joint_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
    ]

    # Random number generator that's used for sampling joint configurations.
    rng = np.random.default_rng(seed=None)

    # Generate valid initial/goal configurations.
    joint_qpos_addrs = utils.joint_names_to_qpos_addrs(joint_names, model)
    lower_limits, upper_limits = utils.joint_limits(joint_names, model)
    q_init = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)
    q_goal = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)

    # Set up the planner.
    # Tweak the values in planner_options to see the effect on generated plans!
    epsilon = 0.05
    planner_options = RRTOptions(
        joint_names=joint_names,
        max_planning_time=10,
        epsilon=epsilon,
        shortcut_filler_epsilon=10*epsilon,
        rng=rng,
        goal_biasing_probability=0.1,
    )
    # The state of MjData passed to the RRT object is used as the initial planning state.
    # random_valid_config modifies MjData in place, so we reset MjData to match q_init before
    # constructing the RRT object.
    update_joint_config(q_init, joint_qpos_addrs, model, data)
    planner = RRT(planner_options, model, data)

    print("Planning...")
    start = time.time()
    path = planner.plan(q_goal)
    if not path:
        print("Planning failed")
        exit()
    print(f"Planning took {time.time() - start}s")

    print("Shortcutting...")
    start = time.time()
    shortcut_path = planner.shortcut(path, num_attempts=len(path))
    print(f"Shortcutting took {time.time() - start}s")

    # Smooth the path by performing naive joint-space B-spline interpolation.
    spline = make_interp_spline(generate_path_timing(path), path)
    spline_shortcut = make_interp_spline(generate_path_timing(shortcut_path), shortcut_path)

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        # Update the viewer's orientation to capture the scene.
        viewer.cam.lookat = [0, 0, 0.35]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25

        # Show the initial EE pose
        update_joint_config(q_init, joint_qpos_addrs, model, data)
        init_pos = data.site(_EE_SITE).xpos
        init_rot = data.site(_EE_SITE).xmat.reshape(3,3)
        viz.add_frame(viewer.user_scn, init_pos, init_rot)

        # Show the target EE pose
        update_joint_config(q_goal, joint_qpos_addrs, model, data)
        target_pos = data.site(_EE_SITE).xpos
        target_rot = data.site(_EE_SITE).xmat.reshape(3,3)
        viz.add_frame(viewer.user_scn, target_pos, target_rot)

        # Show the initial and shortcut paths.
        path_to_color = {
            spline:          [1.0, 0.0, 0.0, 1.0],
            spline_shortcut: [0.0, 1.0, 0.0, 1.0],
        }
        horizon = np.linspace(0, 1, 1000)
        for p, rgba in path_to_color.items():
            for t in horizon:
                q_t = p(t)
                update_joint_config(q_t, joint_qpos_addrs, model, data)
                ee_world_pos = data.site(_EE_SITE).xpos
                viz.add_sphere(viewer.user_scn, ee_world_pos, 0.004, rgba)

        # Ensure the robot is at q_init and then update the viewer to show
        # the frames, paths, and initial state.
        update_joint_config(q_init, joint_qpos_addrs, model, data)
        viewer.sync()

        # Keep the visualizer open until the user closes it.
        # This loop does not need to run at a high rate (doing 10hz here).
        time_between_checks = 1 / 10
        while viewer.is_running():
            time.sleep(time_between_checks)

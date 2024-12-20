'''
Example of how to generate a path and visualize the path waypoints.
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
from mj_maniPlan.sampling import HaltonSampler
import mj_maniPlan.utils as utils
import mj_maniPlan.visualization as viz


_HERE = Path(__file__).parent
_PANDA_XML = _HERE.parent / "models" / "franka_emika_panda" / "scene.xml"

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
    # The finger joints of the gripper are not included here, since those values
    # can stay constant during arm planning.
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
    rng = HaltonSampler(len(joint_names), seed=None)

    # Generate a valid goal configuration.
    print("Generating q_goal...")
    joint_qpos_addrs = utils.joint_names_to_qpos_addrs(joint_names, model)
    lower_limits, upper_limits = utils.joint_limits(joint_names, model)
    q_goal = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)

    # Use the keyframe with ID 0 ("home" in the panda.xml MJCF) as q_init.
    # This also sets MjData.qpos to q_init, which is what we want (the state of MjData that's passed
    # to the RRT object is what's used as the planner's initial state).
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    q_init = data.qpos[joint_qpos_addrs]

    # Set up the planner.
    # Tweak the values in planner_options to see the effect on generated plans!
    planner_options = RRTOptions(
        joint_names=joint_names,
        max_planning_time=10,
        epsilon=0.05,
        rng=rng,
        goal_biasing_probability=0.1,
    )
    planner = RRT(planner_options, model, data)

    print("Planning...")
    start = time.time()
    path = planner.plan(q_goal)
    if not path:
        exit()
    print(f"Planning took {time.time() - start}s")

    print("Shortcutting...")
    start = time.time()
    # TODO: use something like
    # https://github.com/adlarkin/mj_maniPlan/pull/20/commits/c99f2dececd8a6228bd5c6b7fdc704d168b08b12
    # to make sure shortcut_path has enough waypoints for spline fitting?
    # ...
    # OR, after shortcutting, run CONNECT as needed for adjacent waypoints that have a config distance
    # that's greater than RRTOptions.epsilon
    shortcut_path = planner.shortcut(path, num_attempts=int(0.75 * len(path)))
    print(f"Shortcutting took {time.time() - start}s")
    # TODO: remove these print statements once I address the TODO above?
    # The length of the list is no longer a useful metric if I do intermediate CONNECT on the shortcut path
    print(f"Original path length: {len(path)}")
    print(f"Shortcut path length: {len(shortcut_path)}")

    # Smooth the path by performing naive joint-space B-spline interpolation.
    # Note that this may result in waypoints that are in collision.
    spline = make_interp_spline(generate_path_timing(path), path)
    spline_shortcut = make_interp_spline(generate_path_timing(shortcut_path), shortcut_path)

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        # Update the viewer's orientation to capture the arm movement.
        viewer.cam.lookat = [0, 0, 0.35]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25

        # Show the target EE pose (derived from q_goal)
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
        # the target frame, paths, and initial state.
        update_joint_config(q_init, joint_qpos_addrs, model, data)
        viewer.sync()

        # Keep the visualizer open until the user closes it.
        # This loop does not need to run at a high rate (doing 10hz here).
        time_between_checks = 1 / 10
        while viewer.is_running():
            time.sleep(time_between_checks)

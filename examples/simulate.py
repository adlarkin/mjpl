'''
Example of how to generate a path and visualize the path waypoints.
'''

import mujoco
import mujoco.viewer
import numpy as np
import time

from mj_maniPlan.rrt import (
    RRT,
    RRTOptions,
)
from mj_maniPlan.sampling import HaltonSampler
import mj_maniPlan.utils as utils


if __name__ == '__main__':
    model = mujoco.MjModel.from_xml_path('models/franka_emika_panda/scene_with_obstacles.xml')
    data = mujoco.MjData(model)

    # The joints to sample during planning.
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

    # Generate valid initial and goal configurations.
    print("Generating q_init and q_goal...")
    joint_qpos_addrs = utils.joint_names_to_qpos_addrs(joint_names, model)
    lower_limits, upper_limits = utils.joint_limits(joint_names, model)
    q_init = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)
    q_goal = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)

    def set_joint_config(q: np.ndarray):
        data.qpos[joint_qpos_addrs] = q
        mujoco.mj_kinematics(model, data)

    # random_valid_config modifies MjData in-place, so we need to reset the data to q_init before planning.
    set_joint_config(q_init)

    # Example of how to set the joint configuration to the values specified in the keyframe with ID 0.
    # This can be useful if the joint configuration from this keyframe is what's used for q_init.
    '''
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    '''

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
    path = planner.plan(q_goal)
    if not path:
        exit()

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        # Update the viewer's orientation to capture the arm movement.
        viewer.cam.lookat = [0, 0, 0.35]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25

        def set_and_visualize_joint_config(q: np.ndarray):
            set_joint_config(q)
            viewer.sync()

        # Show the start and goal configurations.
        set_and_visualize_joint_config(q_init)
        time.sleep(1.3)
        set_and_visualize_joint_config(q_goal)
        time.sleep(1.25)

        # Visualize kinematic updates at 60hz.
        viz_time_per_frame = 1 / 60

        while viewer.is_running():
            # Visualization of the planner-generated waypoint path.
            for q in path:
                start_time = time.time()
                # Perform a kinematic visualization of the path waypoints.
                # The commented out code block below shows a "hacky" way to perform control along the waypoints.
                set_and_visualize_joint_config(q)
                '''
                # Visualize the plan by sending control signals to the joint actuators.
                data.ctrl[joint_qpos_addrs] = q
                # TODO: figure out what nstep should be (depends on controller hz and simulation dt)
                mujoco.mj_step(model, data, nstep=10)
                viewer.sync()
                '''
                elapsed_time = time.time() - start_time
                if elapsed_time < viz_time_per_frame:
                    time.sleep(viz_time_per_frame - elapsed_time)
            time.sleep(0.25)

import mujoco
import mujoco.viewer
import numpy as np
import time
import rrt


if __name__ == '__main__':
    model = mujoco.MjModel.from_xml_path('models/franka_emika_panda/scene.xml')
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
    rng = np.random.default_rng(seed=None)

    # Generate valid initial and goal configurations.
    joint_qpos_addrs = rrt.joint_names_to_qpos_addrs(joint_names, model)
    lower_limits, upper_limits = rrt.joint_limits(joint_names, model)
    q_init = rrt.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)
    q_goal = rrt.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)

    # is_valid_config modifies MjData in-place, so we need to reset the data to q_init before planning.
    data.qpos[joint_qpos_addrs] = q_init
    mujoco.mj_kinematics(model, data)

    '''
    # Set q_init to the "home" keyframe.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    '''

    # Set up the planner.
    # Tweak the values in planner_options to see the effect on generated plans!
    planner_options = rrt.RRTOptions(
        joint_names=joint_names,
        max_planning_time=10,
        epsilon=0.05,
        rng=rng,
        goal_biasing_probability=0.1,
    )
    planner = rrt.RRT(planner_options, model, data)

    print("Planning...")
    path = planner.plan(q_goal)
    if not path:
        exit()

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        # Update the viewer's orientation to capture the arm movement.
        viewer.cam.lookat = [0.45, 0, 0.3]
        viewer.cam.distance = 1.75
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -10

        # TODO: don't hardcode these? (used to slow down visualization)
        viz_time = 3
        sleep_per_viz_update = viz_time / len(path)

        # Show the start and goal configurations.
        data.qpos[joint_qpos_addrs] = q_init
        mujoco.mj_kinematics(model, data)
        viewer.sync()
        time.sleep(1.5)
        data.qpos[joint_qpos_addrs] = q_goal
        mujoco.mj_kinematics(model, data)
        viewer.sync()
        time.sleep(1.25)
        # After showing q_init and q_goal, reset simulation to q_init before vizualizing the path.
        data.qpos[joint_qpos_addrs] = q_init
        mujoco.mj_kinematics(model, data)

        next_configuration = 0
        while viewer.is_running():
            # Visualize the plan by sending control signals to the joint actuators.
            data.ctrl[joint_qpos_addrs] = path[next_configuration]
            # TODO: figure out what nstep should be (depends on controller hz)
            mujoco.mj_step(model, data, nstep=10)
            viewer.sync()
            next_configuration += 1
            if next_configuration == len(path):
                # reverse the path and move the robot back to its starting state
                next_configuration = 0
                path.reverse()
            time.sleep(sleep_per_viz_update)

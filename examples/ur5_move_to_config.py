import time
from pathlib import Path

import example_utils as ex_utils
import mujoco
import mujoco.viewer
import numpy as np

import mjpl
import mjpl.visualization as viz

_HERE = Path(__file__).parent
_UR5_XML = _HERE / "models" / "universal_robots_ur5e" / "scene.xml"
_UR5_EE_SITE = "attachment_site"


def main():
    visualize, seed = ex_utils.parse_args(
        description="Compute and follow a trajectory to a target configuration."
    )

    model = mujoco.MjModel.from_xml_path(_UR5_XML.as_posix())

    arm_joints = mjpl.all_joints(model)
    q_idx = mjpl.qpos_idx(model, arm_joints)

    cr = mjpl.CollisionRuleset()

    # Let the "home" keyframe in the MJCF be the initial state.
    home_keyframe = model.keyframe("home")
    q_init = home_keyframe.qpos.copy()

    # From the initial state, generate a valid goal configuration.
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
    q_goal = mjpl.random_valid_config(model, q_init, arm_joints, seed, cr)

    # Set up the planner.
    planner = mjpl.RRT(model, arm_joints, cr, seed=seed, goal_biasing_probability=0.1)

    print("Planning...")
    start = time.time()
    path = planner.plan_to_config(q_init, q_goal)
    if not path:
        print("Planning failed")
        return
    print(f"Planning took {(time.time() - start):.4f}s")

    print("Shortcutting...")
    start = time.time()
    shortcut_path = mjpl.shortcut(
        model,
        path,
        cr,
        validation_dist=planner.epsilon,
        max_attempts=len(path.waypoints),
        seed=seed,
    )
    print(f"Shortcutting took {(time.time() - start):.4f}s")

    # The trajectory limits used here are for demonstration purposes only.
    # In practice, consult your hardware spec sheet for this information.
    dof = len(path.waypoints[0])
    traj_generator = mjpl.RuckigTrajectoryGenerator(
        dt=model.opt.timestep,
        max_velocity=np.ones(dof) * np.pi,
        max_acceleration=np.ones(dof) * 0.5 * np.pi,
        max_jerk=np.ones(dof),
    )

    print("Generating trajectory...")
    start = time.time()
    trajectory = mjpl.generate_collision_free_trajectory(
        model, shortcut_path, traj_generator, cr
    )
    print(f"Trajectory generation took {(time.time() - start):.4f}s")

    # Actuator indices in data.ctrl that correspond to the joints in the trajectory.
    actuators = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    actuator_ids = [model.actuator(act).id for act in actuators]

    # Follow the trajectory via position control, starting from the initial state.
    mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
    q_t = [q_init]
    for q_ref in trajectory.positions:
        data.ctrl[actuator_ids] = q_ref
        mujoco.mj_step(model, data)
        q_t.append(data.qpos.copy())

    if visualize:
        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            # Update the viewer's orientation to capture the scene.
            viewer.cam.lookat = [0, 0, 0.35]
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -25

            # Visualize the initial EE pose.
            data.qpos = q_init
            mujoco.mj_kinematics(model, data)
            initial_pose = mjpl.site_pose(data, _UR5_EE_SITE)
            viz.add_frame(
                viewer.user_scn,
                initial_pose.translation(),
                initial_pose.rotation().as_matrix(),
            )

            # Visualize the goal EE pose (derived from the goal config).
            data.qpos[q_idx] = q_goal
            mujoco.mj_kinematics(model, data)
            goal_pose = mjpl.site_pose(data, _UR5_EE_SITE)
            viz.add_frame(
                viewer.user_scn,
                goal_pose.translation(),
                goal_pose.rotation().as_matrix(),
            )

            # Visualize the trajectory. The trajectory is of high resolution,
            # so plotting every other timestep should be sufficient.
            for q_ref in trajectory.positions[::2]:
                data.qpos[q_idx] = q_ref
                mujoco.mj_kinematics(model, data)
                pos = data.site(_UR5_EE_SITE).xpos
                viz.add_sphere(
                    viewer.user_scn, pos, radius=0.004, rgba=[0.2, 0.6, 0.2, 0.2]
                )

            # Replay the robot following the trajectory.
            for q_actual in q_t:
                start_time = time.time()
                if not viewer.is_running():
                    return
                data.qpos = q_actual
                mujoco.mj_kinematics(model, data)
                viewer.sync()
                time_until_next_step = model.opt.timestep - (time.time() - start_time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

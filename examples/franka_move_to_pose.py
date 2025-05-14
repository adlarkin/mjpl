import sys
import time
from pathlib import Path

import example_utils as ex_utils
import mujoco
import mujoco.viewer
import numpy as np

import mjpl
import mjpl.visualization as viz

_HERE = Path(__file__).parent
_PANDA_XML = _HERE / "models" / "franka_emika_panda" / "scene_with_obstacles.xml"
_PANDA_EE_SITE = "ee_site"


# Return whether or not the example was successful. This is used in CI.
def main() -> bool:
    visualize, seed = ex_utils.parse_args(
        description="Compute and follow a trajectory to a goal pose."
    )

    model = mujoco.MjModel.from_xml_path(_PANDA_XML.as_posix())

    # Joints to use during planning. The finger joints are excluded, which means that
    # during planning, the fingers on the gripper should remain "fixed".
    arm_joints = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    q_idx = mjpl.qpos_idx(model, arm_joints)

    # Let the "home" keyframe in the MJCF be the initial state.
    home_keyframe = model.keyframe("home")
    q_init = home_keyframe.qpos.copy()

    allowed_collisions = [("left_finger", "right_finger")]
    constraints = [
        mjpl.JointLimitConstraint(model),
        mjpl.CollisionConstraint(model, allowed_collisions),
    ]

    # From the initial state, generate a valid goal pose that's derived from a
    # valid joint configuration.
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
    data.qpos = mjpl.random_config(model, q_init, arm_joints, seed, constraints)
    mujoco.mj_kinematics(model, data)
    goal_pose = mjpl.site_pose(data, _PANDA_EE_SITE)

    # Set up the planner.
    planner = mjpl.RRT(
        model, arm_joints, constraints, seed=seed, goal_biasing_probability=0.1
    )

    print("Planning...")
    start = time.time()
    waypoints = planner.plan_to_pose(q_init, goal_pose, _PANDA_EE_SITE)
    if not waypoints:
        print("Planning failed")
        return False
    print(f"Planning took {(time.time() - start):.4f}s")

    print("Shortcutting...")
    start = time.time()
    shortcut_waypoints = mjpl.smooth_path(
        waypoints, constraints, eps=planner.epsilon, seed=seed
    )
    print(f"Shortcutting took {(time.time() - start):.4f}s")

    # The trajectory limits used here are for demonstration purposes only.
    # In practice, consult your hardware spec sheet for this information.
    dof = len(waypoints[0])
    traj_generator = mjpl.ToppraTrajectoryGenerator(
        dt=model.opt.timestep,
        max_velocity=np.ones(dof) * 0.5 * np.pi,
        max_acceleration=np.ones(dof) * 0.25 * np.pi,
    )

    print("Generating trajectory...")
    start = time.time()
    trajectory = traj_generator.generate_trajectory(shortcut_waypoints)
    if trajectory is None:
        print("Trajectory generation failed.")
        return False
    print(f"Trajectory generation took {(time.time() - start):.4f}s")

    # Follow the trajectory via position control, starting from the initial state.
    # Send position commands to the arm actuators since planning was done for the arm
    # joints (ignore the last actuator, which corresponds to the gripper fingers).
    mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
    q_t = [q_init]
    for q_ref in trajectory.positions:
        data.ctrl[:-1] = q_ref[q_idx]
        mujoco.mj_step(model, data)
        q_t.append(data.qpos.copy())

    if visualize:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            assert viewer.user_scn is not None

            # Update the viewer's orientation to capture the scene.
            viewer.cam.lookat = [0, 0, 0.35]
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -25

            # Visualize the initial EE pose.
            data.qpos = q_init
            mujoco.mj_kinematics(model, data)
            initial_pose = mjpl.site_pose(data, _PANDA_EE_SITE)
            viz.add_frame(
                viewer.user_scn,
                initial_pose.translation(),
                initial_pose.rotation().as_matrix(),
            )

            # Visualize the goal EE pose.
            viz.add_frame(
                viewer.user_scn,
                goal_pose.translation(),
                goal_pose.rotation().as_matrix(),
            )

            # Visualize the trajectory. The trajectory is of high resolution,
            # so plotting every other timestep should be sufficient.
            for q_ref in trajectory.positions[::2]:
                data.qpos = q_ref
                mujoco.mj_kinematics(model, data)
                pos = data.site(_PANDA_EE_SITE).xpos
                viz.add_sphere(
                    viewer.user_scn,
                    pos,
                    radius=0.004,
                    rgba=np.array([0.2, 0.6, 0.2, 0.2]),
                )

            # Replay the robot following the trajectory.
            for q_actual in q_t:
                start_time = time.time()
                if not viewer.is_running():
                    return True
                data.qpos = q_actual
                mujoco.mj_kinematics(model, data)
                viewer.sync()
                time_until_next_step = model.opt.timestep - (time.time() - start_time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

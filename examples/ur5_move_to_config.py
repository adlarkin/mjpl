import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from example_utils import parse_args

import mj_maniPlan.utils as utils
import mj_maniPlan.visualization as viz
from mj_maniPlan.collision_ruleset import CollisionRuleset
from mj_maniPlan.joint_group import JointGroup
from mj_maniPlan.rrt import RRT, RRTOptions
from mj_maniPlan.trajectory.ruckig_trajectory import RuckigTrajectoryGenerator

_HERE = Path(__file__).parent
_UR5_XML = _HERE / "models" / "universal_robots_ur5e" / "scene.xml"
_UR5_EE_SITE = "attachment_site"


def main():
    visualize, seed = parse_args(
        description="Compute and follow a trajectory to a target configuration."
    )

    model = mujoco.MjModel.from_xml_path(_UR5_XML.as_posix())

    arm_joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    arm_joint_ids = [model.joint(joint).id for joint in arm_joints]
    arm_jg = JointGroup(model, arm_joint_ids)

    cr = CollisionRuleset(model)

    # Let the "home" keyframe in the MJCF be the initial state.
    home_keyframe = model.keyframe("home")
    q_init = home_keyframe.qpos.copy()

    # From the initial state, generate a valid goal configuration.
    rng = np.random.default_rng(seed=seed)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
    q_goal = utils.random_valid_config(rng, arm_jg, data, cr)

    # Set up the planner.
    planner_options = RRTOptions(
        jg=arm_jg,
        cr=cr,
        max_planning_time=10.0,
        epsilon=0.05,
        seed=seed,
        goal_biasing_probability=0.1,
        max_connection_distance=np.inf,
    )
    planner = RRT(planner_options)

    print("Planning...")
    start = time.time()
    path = planner.plan_to_config(q_init, q_goal)
    if not path:
        print("Planning failed")
        return
    print(f"Planning took {(time.time() - start):.4f}s")

    print("Shortcutting...")
    start = time.time()
    shortcut_path = utils.shortcut(
        path,
        arm_jg,
        model,
        cr,
        validation_dist=planner_options.epsilon,
        max_attempts=len(path),
        seed=seed,
    )
    print(f"Shortcutting took {(time.time() - start):.4f}s")

    # The trajectory limits used here are for demonstration purposes only.
    # In practice, consult your hardware spec sheet for this information.
    dof = len(arm_joints)
    traj_generator = RuckigTrajectoryGenerator(
        dt=model.opt.timestep,
        max_velocity=np.ones(dof) * np.pi,
        max_acceleration=np.ones(dof) * 0.5 * np.pi,
        max_jerk=np.ones(dof),
    )

    print("Generating trajectory...")
    start = time.time()
    trajectory = traj_generator.generate_trajectory(shortcut_path)
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
        # While the planner gives a sequence of waypoints are collision free, the
        # generated trajectory may not. For more info, see:
        # https://github.com/adlarkin/mj_maniPlan/issues/54
        if not cr.obeys_ruleset(data.contact.geom):
            print("Invalid collision occurred during trajectory execution.")
            return
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
            initial_pose = utils.site_pose(data, _UR5_EE_SITE)
            viz.add_frame(
                viewer.user_scn,
                initial_pose.translation(),
                initial_pose.rotation().as_matrix(),
            )

            # Visualize the goal EE pose (derived from the goal config).
            arm_jg.fk(q_goal, data)
            goal_pose = utils.site_pose(data, _UR5_EE_SITE)
            viz.add_frame(
                viewer.user_scn,
                goal_pose.translation(),
                goal_pose.rotation().as_matrix(),
            )

            # Visualize the trajectory. The trajectory is of high resolution,
            # so plotting every other timestep should be sufficient.
            for q_ref in trajectory.positions[::2]:
                arm_jg.fk(q_ref, data)
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

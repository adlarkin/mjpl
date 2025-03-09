import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from example_utils import parse_args

import mj_maniPlan.visualization as viz
from mj_maniPlan.collision_ruleset import CollisionRuleset
from mj_maniPlan.inverse_kinematics import IKOptions
from mj_maniPlan.joint_group import JointGroup
from mj_maniPlan.rrt import RRT, RRTOptions
from mj_maniPlan.trajectory import TrajectoryLimits, generate_trajectory
from mj_maniPlan.utils import random_valid_config

_HERE = Path(__file__).parent
_PANDA_XML = _HERE / "models" / "franka_emika_panda" / "scene_with_obstacles.xml"
_PANDA_EE_SITE = "ee_site"


def main():
    visualize, seed = parse_args(
        description="Compute and follow a trajectory to a target pose."
    )

    model = mujoco.MjModel.from_xml_path(_PANDA_XML.as_posix())

    arm_joints = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    arm_joint_ids = [model.joint(joint).id for joint in arm_joints]
    arm_jg = JointGroup(model, arm_joint_ids)

    q_init = model.keyframe("home").qpos.copy()

    # Generate a valid target pose that's derived from a valid joint configuration.
    rng = np.random.default_rng(seed=seed)
    data = mujoco.MjData(model)
    q_rand = random_valid_config(rng, arm_jg, data, CollisionRuleset(model))
    arm_jg.fk(q_rand, data)
    target_pos = data.site(_PANDA_EE_SITE).xpos.copy()
    target_rotmat = data.site(_PANDA_EE_SITE).xmat.copy()

    allowed_collisions = np.array(
        [
            [model.body("left_finger").id, model.body("right_finger").id],
        ]
    )
    cr = CollisionRuleset(model, allowed_collisions)

    # Set up the planner.
    epsilon = 0.05
    planner_options = RRTOptions(
        jg=arm_jg,
        cr=cr,
        max_planning_time=10,
        epsilon=epsilon,
        seed=seed,
        goal_biasing_probability=0.1,
        max_connection_distance=np.inf,
    )
    ik_options = IKOptions(
        jg=arm_jg,
        cr=cr,
        seed=seed,
        max_attempts=5,
    )
    planner = RRT(planner_options)

    print("Planning...")
    start = time.time()
    path = planner.plan_to_pose(
        q_init, _PANDA_EE_SITE, target_pos, target_rotmat.reshape(3, 3), ik_options
    )
    if not path:
        print("Planning failed")
        return
    print(f"Planning took {(time.time() - start):.4f}s")

    print("Shortcutting...")
    start = time.time()
    shortcut_path = planner.shortcut(path, np.inf, num_attempts=len(path))
    print(f"Shortcutting took {(time.time() - start):.4f}s")

    # These values are for demonstration purposes only.
    # In practice, consult your hardware spec sheet for this information.
    dof = len(arm_joints)
    tr_limits = TrajectoryLimits(
        jg=arm_jg,
        min_velocity=-np.ones(dof) * np.pi,
        max_velocity=np.ones(dof) * np.pi,
        min_acceleration=-np.ones(dof) * 0.5 * np.pi,
        max_acceleration=np.ones(dof) * 0.5 * np.pi,
        jerk=np.ones(dof),
    )

    print("Generating trajectory...")
    start = time.time()
    traj = generate_trajectory(shortcut_path, tr_limits, model.opt.timestep)
    print(f"Trajectory generation took {(time.time() - start):.4f}s")

    # Actuator indices in data.ctrl that correspond to the joints in the trajectory.
    actuators = [
        "actuator1",
        "actuator2",
        "actuator3",
        "actuator4",
        "actuator5",
        "actuator6",
        "actuator7",
    ]
    actuator_ids = [model.actuator(act).id for act in actuators]

    # Follow the trajectory via position control, starting from the initial state.
    data.qpos = q_init.copy()
    mujoco.mj_forward(model, data)
    q_t = [arm_jg.qpos(data)]
    for q_ref in traj.configurations:
        data.ctrl[actuator_ids] = q_ref
        mujoco.mj_step(model, data)
        q_t.append(arm_jg.qpos(data))

    if visualize:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Update the viewer's orientation to capture the scene.
            viewer.cam.lookat = [0, 0, 0.35]
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -25

            # Visualize the initial EE pose.
            arm_jg.fk(q_t[0], data)
            mujoco.mj_kinematics(model, data)
            site = data.site(_PANDA_EE_SITE)
            viz.add_frame(viewer.user_scn, site.xpos, site.xmat.reshape(3, 3))

            # Visualize the target EE pose.
            viz.add_frame(viewer.user_scn, target_pos, target_rotmat.reshape(3, 3))

            # Visualize the trajectory. The trajectory is of high resolution,
            # so plotting every other timestep should be sufficient.
            for q_ref in traj.configurations[::2]:
                arm_jg.fk(q_ref, data)
                pos = data.site(_PANDA_EE_SITE).xpos
                viz.add_sphere(viewer.user_scn, pos, 0.004, [0.2, 0.6, 0.2, 0.2])

            # Replay the robot following the trajectory.
            for q_actual in q_t:
                start_time = time.time()
                if not viewer.is_running():
                    return
                arm_jg.fk(q_actual, data)
                viewer.sync()
                time_until_next_step = model.opt.timestep - (time.time() - start_time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

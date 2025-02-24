"""
Example of the planning -> trajectory generation -> trajectory following pipeline.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import mj_maniPlan.utils as utils
import mj_maniPlan.visualization as viz
from mj_maniPlan.collision_ruleset import CollisionRuleset
from mj_maniPlan.joint_group import JointGroup
from mj_maniPlan.rrt import (
    RRT,
    RRTOptions,
)
from mj_maniPlan.trajectory import TrajectoryLimits, generate_trajectory

_HERE = Path(__file__).parent
_PANDA_XML = _HERE / "models" / "franka_emika_panda" / "scene.xml"
_PANDA_OBSTACLES_XML = (
    _HERE / "models" / "franka_emika_panda" / "scene_with_obstacles.xml"
)
_PANDA_EE_SITE = "ee_site"


def parse_args() -> tuple[bool, bool, int | None]:
    parser = argparse.ArgumentParser(description="Compute and follow a trajectory.")
    parser.add_argument(
        "-viz",
        "--visualize",
        action="store_true",  # set to True if flag is provided
        default=False,  # default value if flag is not provided
        help="Visualize paths via the mujoco viewer",
    )
    parser.add_argument(
        "-obs",
        "--obstacles",
        action="store_true",  # set to True if flag is provided
        default=False,  # default value if flag is not provided
        help="Use obstacles in the environment",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=-1,
        help="Seed for random sampling. Must be >= 0. If not set, a random seed will be used",
    )
    args = parser.parse_args()
    seed = args.seed
    if seed < 0:
        seed = None
    return args.visualize, args.obstacles, seed


def main():
    visualize, use_obstacles, seed = parse_args()

    if use_obstacles:
        model = mujoco.MjModel.from_xml_path(_PANDA_OBSTACLES_XML.as_posix())
    else:
        model = mujoco.MjModel.from_xml_path(_PANDA_XML.as_posix())
    data = mujoco.MjData(model)

    joint_ids = [
        model.joint("joint1").id,
        model.joint("joint2").id,
        model.joint("joint3").id,
        model.joint("joint4").id,
        model.joint("joint5").id,
        model.joint("joint6").id,
        model.joint("joint7").id,
    ]
    jg = JointGroup(joint_ids, model)

    allowed_collisions = [
        (model.body("left_finger").id, model.body("right_finger").id),
    ]
    cr = CollisionRuleset(model, allowed_collisions)

    # Use the 'home' keyframe position as q_init.
    world_q_init = model.keyframe("home").qpos

    # Generate valid goal configuration.
    rng = np.random.default_rng(seed=seed)
    q_goal = utils.random_valid_config(rng, jg, data, cr)

    # Set up the planner.
    epsilon = 0.05
    planner_options = RRTOptions(
        jg=jg,
        cr=cr,
        max_planning_time=10,
        epsilon=epsilon,
        shortcut_filler_epsilon=10 * epsilon,
        seed=seed,
        goal_biasing_probability=0.1,
    )
    planner = RRT(planner_options)

    print("Planning...")
    start = time.time()
    path = planner.plan(world_q_init, q_goal)
    if not path:
        print("Planning failed")
        return
    print(f"Planning took {(time.time() - start):.4f}s")

    print("Shortcutting...")
    start = time.time()
    shortcut_path = planner.shortcut(path, num_attempts=len(path))
    print(f"Shortcutting took {(time.time() - start):.4f}s")

    tr_limits = TrajectoryLimits(
        jg=jg,
        velocity=np.ones(len(jg.joints())) * np.pi,
        acceleration=np.ones(len(jg.joints())) * 0.5 * np.pi,
        jerk=np.ones(len(jg.joints())),
    )

    print("Generating trajectory...")
    start = time.time()
    traj = generate_trajectory(shortcut_path, tr_limits, model.opt.timestep)
    print(f"Trajectory generation took {(time.time() - start):.4f}s")

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
            data.qpos = world_q_init
            mujoco.mj_kinematics(model, data)
            pos, rot = utils.site_pose(_PANDA_EE_SITE, data)
            viz.add_frame(viewer.user_scn, pos, rot)

            # Visualize the target EE pose.
            jg.fk(q_goal, data)
            pos, rot = utils.site_pose(_PANDA_EE_SITE, data)
            viz.add_frame(viewer.user_scn, pos, rot)

            # Visualize the trajectory. The trajectory is of high resolution,
            # so plotting every other timestep should be sufficient.
            for q_t in traj.configurations[::2]:
                jg.fk(q_t, data)
                pos = data.site(_PANDA_EE_SITE).xpos
                viz.add_sphere(viewer.user_scn, pos, 0.004, [0.2, 0.6, 0.2, 0.2])

            # Make sure MjData matches the initial planning world state and then
            # update the viewer to show the frames and trajectory.
            data.qpos = world_q_init
            mujoco.mj_kinematics(model, data)
            viewer.sync()

            # Actuator indices in data.ctrl
            actuator_ids = [
                model.actuator("actuator1").id,
                model.actuator("actuator2").id,
                model.actuator("actuator3").id,
                model.actuator("actuator4").id,
                model.actuator("actuator5").id,
                model.actuator("actuator6").id,
                model.actuator("actuator7").id,
            ]

            # Command the robot along the trajectory via position control.
            while viewer.is_running():
                time.sleep(0.5)

                mujoco.mj_resetData(model, data)
                mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)

                for q_ref in traj.configurations:
                    start_time = time.time()
                    data.ctrl[actuator_ids] = q_ref
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time_until_next_step = model.opt.timestep - (
                        time.time() - start_time
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

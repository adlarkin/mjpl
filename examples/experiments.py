import mink
import mujoco
import mujoco.viewer
import numpy as np
import os
import pickle
import time
from scipy.interpolate import make_interp_spline

from mj_maniPlan.rrt import (
    RRT,
    RRTOptions,
)
from mj_maniPlan.sampling import HaltonSampler
import mj_maniPlan.utils as utils


def cartesian_distance(p1: np.ndarray, p2: np.ndarray):
    return np.linalg.norm(p2 - p1)

def get_EE_world_position(config: mink.Configuration):
    ee_pose = config.get_transform_frame_to_world(
        frame_name="ee_site",
        frame_type="site",
    )
    return ee_pose.translation()

# Naive timing generation for a path, which can be used for something like B-spline interpolation.
# Configuration distance between two adjacent path waypoints - q_curr, q_next - is used as a notion
# for the time it takes to move from q_curr to q_next.
def generate_path_timing(path):
    timing = [0.0]
    for i in range(1, len(path)):
        timing.append(timing[-1] + utils.configuration_distance(path[i-1], path[i]))
    # scale to [0..1]
    return np.interp(timing, (timing[0], timing[-1]), (0, 1))


if __name__ == "__main__":
    # Data that's needed for plotting.
    # The solution_{times, lengths} lists are the same length for IK and RRT, respectively.
    # These lists will be used for box plots.
    # A bar plot can be used to show the failed attempts of IK and RRT.
    plot_data = {
        'IK_valid_solution_times': [],
        'IK_valid_solution_lengths': [],
        'IK_success_rate': 0.0,

        'RRT_valid_solution_times': [],
        'RRT_valid_solution_lengths': [],
        'RRT_success_rate': 0.0,

        'RRT_shortcutting_valid_solution_times': [],
        'RRT_shortcutting_valid_solution_lengths': [],
        # num_of_failed attempts isn't needed for shortcutting b/c we don't shortcut if RRT fails
    }

    dir = os.path.dirname(os.path.realpath(__file__))
    model_xml_path = dir + "/../models/franka_emika_panda/scene.xml"
    model = mujoco.MjModel.from_xml_path(model_xml_path)
    data = mujoco.MjData(model)

    # Joints used in planning.
    joint_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
    ]

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="ee_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # TODO: load the scene with obstacles into MjModel and then add obstacle avoidance here
    hand_geoms = mink.get_subtree_geom_ids(model, model.body("hand").id)
    collision_pairs = [
        # TODO: add collision avoidance between the hand and rest of the robot?
        (hand_geoms, ["floor"]),
    ]
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    max_velocities = { j: np.pi for j in joint_names }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    joint_qpos_addrs = utils.joint_names_to_qpos_addrs(joint_names, model)

    # Save initial joint config as the home keyframe.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    q_init = data.qpos[joint_qpos_addrs].copy()

    # IK error threshold when checking if the target EE pose has been reached.
    pos_threshold = 1e-3
    ori_threshold = 1e-3

    # Timeout for a planning attempt, in seconds.
    timeout = 5

    num_runs = 100
    for s in range(num_runs):
        configuration = mink.Configuration(model)

        # Create a sampler with a particular seed.
        rng = HaltonSampler(len(max_velocities), seed=s)

        # Randomly sample a valid joint config for the goal.
        lower_limits, upper_limits = utils.joint_limits(joint_names, model)
        q_goal = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)

        '''
        Start of IK experiments
        '''
        # For IK planning, we need an EE pose.
        # This is extracted from q_goal.
        data.qpos[joint_qpos_addrs] = q_goal
        configuration.update(data.qpos)
        ee_target_pose = configuration.get_transform_frame_to_world(
            frame_name="ee_site",
            frame_type="site",
        )

        # Initialize the IK problem to start at q_init.
        data.qpos[joint_qpos_addrs] = q_init
        configuration.update(data.qpos)

        # move the robot from q_init to q_goal by solving IK
        end_effector_task.set_target(ee_target_pose)
        prev_EE_position = get_EE_world_position(configuration)
        solution_len = 0
        start_time = time.time()
        while True:
            vel = mink.solve_ik(
                configuration, tasks, model.opt.timestep, solver="quadprog", damping=1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, model.opt.timestep)
            next_EE_position = get_EE_world_position(configuration)
            solution_len += cartesian_distance(prev_EE_position, next_EE_position)
            prev_EE_position = next_EE_position
            err = end_effector_task.compute_error(configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
            if pos_achieved and ori_achieved:
                plot_data['IK_valid_solution_times'].append(time.time() - start_time)
                plot_data['IK_valid_solution_lengths'].append(solution_len)
                break
            if (time.time() - start_time) > timeout:
                plot_data['IK_success_rate'] += 1
                break
        '''
        End of IK experiemnts
        '''

        '''
        Start of RRT experiments
        '''
        # Start planning at q_init
        data.qpos[joint_qpos_addrs] = q_init
        mujoco.mj_forward(model, data)

        planner_options = RRTOptions(
            joint_names=joint_names,
            max_planning_time=timeout,
            epsilon=0.05,
            rng=rng,
            goal_biasing_probability=0.1,
        )
        planner = RRT(planner_options, model, data)

        start_time = time.time()
        path = planner.plan(q_goal)
        if path:
            plot_data['RRT_valid_solution_times'].append(time.time() - start_time)

            shortened_path = planner.shortcut(path, num_attempts=len(path))
            plot_data['RRT_shortcutting_valid_solution_times'].append(time.time() - start_time)

            def compute_path_len(p):
                timing = generate_path_timing(p)
                spline = make_interp_spline(timing, p)
                data.qpos[joint_qpos_addrs] = q_init
                configuration.update(data.qpos)
                prev_EE_position = get_EE_world_position(configuration)
                solution_len = 0
                horizon = np.linspace(0, 1, int(1 / model.opt.timestep))
                for t in horizon:
                    data.qpos[joint_qpos_addrs] = spline(t)
                    configuration.update(data.qpos)
                    next_EE_position = get_EE_world_position(configuration)
                    solution_len += cartesian_distance(prev_EE_position, next_EE_position)
                    prev_EE_position = next_EE_position
                return solution_len

            plot_data['RRT_valid_solution_lengths'].append(compute_path_len(path))
            plot_data['RRT_shortcutting_valid_solution_lengths'].append(compute_path_len(shortened_path))
        else:
            plot_data['RRT_success_rate'] += 1
        '''
        End of RRT experiments
        '''

    # at the moment, plot_data's success_rate fields have the # of failed runs.
    # convert this to the actual success rate
    IK_failed_runs = plot_data['IK_success_rate']
    plot_data['IK_success_rate'] = (num_runs - IK_failed_runs) / num_runs
    RRT_failed_runs = plot_data['RRT_success_rate']
    plot_data['RRT_success_rate'] = (num_runs - RRT_failed_runs) / num_runs

    with open('data.pickle', 'wb') as f:
        pickle.dump(plot_data, f)

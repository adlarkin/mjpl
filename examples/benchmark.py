'''
Benchmark for testing planning time.

The testing methodology is as follows:
1. To be deterministic, seed the random number generator.
2. Create a planner and a [q_init, q_goal] pairing.
3. Plan from q_init to q_goal N number of times.
4. Report the following:
    a. How many plans succeeded vs how many plans timed out (success rate)
    b. Median planning time of successful planning attempts
'''

import mujoco
import numpy as np
import time
from pathlib import Path

from mj_maniPlan.rrt import (
    RRT,
    RRTOptions,
)
from mj_maniPlan.sampling import HaltonSampler
import mj_maniPlan.utils as utils


_HERE = Path(__file__).parent
_PANDA_XML = _HERE.parent / "models" / "franka_emika_panda" / "scene.xml"


if __name__ == '__main__':
    # NOTE: modify these parameters as needed for your benchmarking needs.
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
    max_planning_time = 10
    epsilon = 0.05
    #seed = 5
    seed = 42
    goal_biasing_probability = 0.1
    number_of_attempts = 15

    model = mujoco.MjModel.from_xml_path(_PANDA_XML.as_posix())

    joint_qpos_addrs = utils.joint_names_to_qpos_addrs(joint_names, model)
    lower_limits, upper_limits = utils.joint_limits(joint_names, model)

    # Plan number_of_attempts times and record benchmarks.
    successful_planning_times = []
    for i in range(number_of_attempts):
        data = mujoco.MjData(model)

        # Random number generator that's used for sampling joint configurations.
        rng = HaltonSampler(len(joint_names), seed=seed)

        # Generate a valid initial and goal configuration.
        q_init = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)
        q_goal = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)

        # set robot joint configuration to q_init
        data.qpos[joint_qpos_addrs] = q_init
        mujoco.mj_kinematics(model, data)

        planner_options = RRTOptions(
            joint_names=joint_names,
            max_planning_time=max_planning_time,
            epsilon=epsilon,
            rng=rng,
            goal_biasing_probability=goal_biasing_probability,
        )
        planner = RRT(planner_options, model, data)

        print(f"Attempt {i}...")
        start_time = time.time()
        path = planner.plan(q_goal)
        elapsed_time = time.time() - start_time
        if path:
            successful_planning_times.append(elapsed_time)
    print()

    successful_attempts = len(successful_planning_times)
    success_rate = successful_attempts / number_of_attempts
    median_planning_time = np.median(successful_planning_times)
    print(f"Attempted {number_of_attempts} plans, succeeded on {successful_attempts} attempts (success rate of {success_rate})")
    print(f"Median planning time of successful plans: {median_planning_time} seconds")

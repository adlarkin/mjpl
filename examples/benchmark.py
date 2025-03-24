"""
Benchmark for testing planning time.

The testing methodology is as follows:
1. To be deterministic, seed the random number generator.
2. Create a planner and a [q_init, ee_goal_pose] pairing.
3. Plan from q_init to ee_goal_pose N number of times.
4. Report the following:
    a. How many plans succeeded vs how many plans timed out (success rate)
    b. Median planning time of successful planning attempts
"""

import time
from pathlib import Path

import mujoco
import numpy as np

import mj_maniPlan.utils as utils
from mj_maniPlan.collision_ruleset import CollisionRuleset
from mj_maniPlan.joint_group import JointGroup
from mj_maniPlan.rrt import RRT, RRTOptions

_HERE = Path(__file__).parent
_PANDA_XML = _HERE / "models" / "franka_emika_panda" / "scene.xml"
_PANDA_EE_SITE = "ee_site"


if __name__ == "__main__":
    # NOTE: modify these parameters as needed for your benchmarking needs.
    model = mujoco.MjModel.from_xml_path(_PANDA_XML.as_posix())
    planning_joints = [
        model.joint("joint1").id,
        model.joint("joint2").id,
        model.joint("joint3").id,
        model.joint("joint4").id,
        model.joint("joint5").id,
        model.joint("joint6").id,
        model.joint("joint7").id,
    ]
    allowed_collisions = np.array(
        [
            [model.body("left_finger").id, model.body("right_finger").id],
        ]
    )
    max_planning_time = 10
    epsilon = 0.05
    seed = 42
    goal_biasing_probability = 0.1
    number_of_attempts = 15

    arm_jg = JointGroup(model, planning_joints)
    cr = CollisionRuleset(model, allowed_collisions)

    # Plan number_of_attempts times and record benchmarks.
    successful_planning_times = []
    for i in range(number_of_attempts):
        # Let the "home" keyframe in the MJCF be the initial state.
        home_keyframe = model.keyframe("home")
        q_init = home_keyframe.qpos.copy()

        # From the initial state, generate a goal pose.
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
        rng = np.random.default_rng(seed=seed)
        q_goal = utils.random_valid_config(rng, arm_jg, data, cr)
        arm_jg.fk(q_goal, data)
        goal_pose = utils.site_pose(data, _PANDA_EE_SITE)

        planner_options = RRTOptions(
            jg=arm_jg,
            cr=cr,
            max_planning_time=max_planning_time,
            epsilon=epsilon,
            seed=seed,
            goal_biasing_probability=goal_biasing_probability,
        )
        planner = RRT(planner_options)

        print(f"Attempt {i}...")
        start_time = time.time()
        path = planner.plan_to_pose(q_init, goal_pose, _PANDA_EE_SITE)
        elapsed_time = time.time() - start_time
        if path:
            successful_planning_times.append(elapsed_time)
    print()

    successful_attempts = len(successful_planning_times)
    success_rate = successful_attempts / number_of_attempts
    median_planning_time = np.median(successful_planning_times)
    print(
        f"Attempted {number_of_attempts} plans, succeeded on {successful_attempts} attempts (success rate of {success_rate:.2f})"
    )
    print(
        f"Median planning time of successful plans: {median_planning_time:.4f} seconds"
    )

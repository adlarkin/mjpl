import mujoco
import mujoco.viewer
import time

from rrt import RRT


if __name__ == '__main__':
    model = mujoco.MjModel.from_xml_path('models/franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)

    # assumes that a keyframe exists
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)

    # generate a goal configuration and build a plan
    planner = RRT(model, data)
    q_goal = planner.random_config()
    print(f"q_goal is {q_goal}")
    path = planner.build_tree(q_goal)

    if path:
        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
            # update the viewer's orientation to capture the arm movement
            viewer.cam.lookat = [0.45, 0, 0.3]
            viewer.cam.distance = 1.75
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -10

            while viewer.is_running():
                # show initial state
                data.qpos[:7] = path[0]
                mujoco.mj_kinematics(model, data)
                viewer.sync()
                time.sleep(1)
                # show goal state
                data.qpos[:7] = path[-1]
                mujoco.mj_kinematics(model, data)
                viewer.sync()
                time.sleep(1)
                # show path
                # TODO: don't hardcode this (temporary for path visualization)
                viz_time = 3
                sleep_per_step = viz_time / len(path)
                for q in path:
                    data.qpos[:7] = q
                    mujoco.mj_kinematics(model, data)
                    viewer.sync()
                    time.sleep(sleep_per_step)
                time.sleep(0.25)

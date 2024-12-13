import mujoco
import mujoco.viewer
import numpy as np
import os
import pickle
import time


# Function to add a coordinate frame to the scene
def add_frame(scene: mujoco.MjvScene, position, rot_mat, axis_radius=0.0075, axis_half_length=.0525):
    # Axis colors (RGBA): X (red), Y (green), Z (blue)
    colors = {
        'x': [1.0, 0.0, 0.0, 1.0],
        'y': [0.0, 1.0, 0.0, 1.0],
        'z': [0.0, 0.0, 1.0, 1.0]
    }
    # Define axis directions
    axes = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0])
    }

    for axis, color in colors.items():
        # Rotate axis direction by the orientation
        direction = rot_mat @ axes[axis]
        # Set the end point of the axis marker
        end_point = position + axis_half_length * direction

        assert scene.ngeom < scene.maxgeom
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array(color),
        )
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            axis_radius,
            position,
            end_point,
        )

def visualize_path(scene: mujoco.MjvScene, path, marker_radius, rgba):
    for pos in path:
        assert scene.ngeom < scene.maxgeom
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom-1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[marker_radius, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )


if __name__ == "__main__":
    dir = os.path.dirname(os.path.realpath(__file__))
    model_xml_path = dir + "/../models/franka_emika_panda/scene.xml"
    model = mujoco.MjModel.from_xml_path(model_xml_path)
    data = mujoco.MjData(model)

    with open('data.pickle', 'rb') as f:
        path_data = pickle.load(f)
    path_data = path_data['valid_paths']

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        # Update the viewer's orientation to capture the arm movement.
        viewer.cam.lookat = [0, 0, 0.35]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25

        while viewer.is_running():
            # disable shadows so it's easier to see the paths
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

            path_marker_radius = 0.005
            for valid_path in path_data:
                print(f"visualizing path from seed {valid_path['seed']}")

                # "Reset" the scene before rendering any paths.
                # This is done by making all current scene geometries transparent.
                for i in range(viewer.user_scn.maxgeom):
                    viewer.user_scn.geoms[i].rgba = np.zeros(4)
                viewer.user_scn.ngeom = 0

                data.qpos = valid_path['q_init']
                mujoco.mj_kinematics(model, data)

                target_pose = valid_path['EE_goal_pose']
                target_position = target_pose.translation()
                target_rotation = target_pose.rotation().as_matrix()
                add_frame(viewer.user_scn, target_position, target_rotation, 0.015, .105)

                visualize_path(viewer.user_scn, valid_path['IK_path'], path_marker_radius, [1, 0, 0, 1])
                visualize_path(viewer.user_scn, valid_path['RRT_path'], path_marker_radius, [0, 1, 0, 1])
                visualize_path(viewer.user_scn, valid_path['RRT_shortcut_path'], path_marker_radius, [0, 0, 1, 1])

                viewer.sync()

                # TODO: replace this with a key_callback so that I can step through each run manually (maybe n for next and p for prev)
                time.sleep(2)

import mujoco
import numpy as np


# Function to add a coordinate frame to the scene.
#
# origin is a vector in R^3 that gives the (x,y,z) position of the frame origin.
# rot_mat is a 3x3 rotation matrix that specifies the frame orientation.
# Both origin and rot_mat are assumed to be defined in the world frame.
def add_frame(scene: mujoco.MjvScene, origin, rot_mat, axis_radius=0.0075, axis_length=.0525):
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
        # Set the end point of the axis marker.
        end_point = origin + axis_length * direction

        assert scene.ngeom < scene.maxgeom
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=color,
            # TODO: remove this or make color axes above np.array type?
            #rgba=np.array(color),
        )
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            width=axis_radius,
            from_=origin,
            to=end_point,
        )

def add_sphere(scene: mujoco.MjvScene, origin, radius, rgba):
    assert scene.ngeom < scene.maxgeom
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom-1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=origin,
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )

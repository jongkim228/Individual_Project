import numpy as np
import mujoco

def calculate_in_local(model, data, camera_name, cube_id):
    mujoco.mj_forward(model, data)
    camera_id = model.camera(camera_name).id
    world_cam = data.cam_xpos[camera_id]
    world_obj = data.xpos[cube_id]

    camera_rot = data.cam_xmat[camera_id].reshape(3,3)

    local_distance = camera_rot.T @ (world_obj - world_cam)

    return local_distance

def objects_in_fov(model,local,camera_name,height,width):
    x, y, z = local
    if z >= 0:
        return False

    zz = -z
    cam_id = model.camera(camera_name).id
    vfov_deg = np.deg2rad(model.cam_fovy[cam_id]) 
    tan_v = np.tan(vfov_deg / 2)

    aspect = width / height

    tan_h = tan_v * aspect

    nx = x / zz
    ny = y / zz

    if abs(nx) > tan_h:
        return False
    
    if abs(ny) > tan_v:
        return False
    
    return True
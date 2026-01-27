import numpy as np
import mujoco
import mujoco.viewer
import time
import cv2
import math

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

arm_actuator_names = [
    "actuator1", "actuator2", "actuator3",
    "actuator4", "actuator5", "actuator6", "actuator7"
]


arm_actuator_ids = np.array([model.actuator(name).id for name in arm_actuator_names])
gripper_id = model.actuator("actuator8").id


hand_id = model.body("hand").id
hand_pos = data.xpos[hand_id]

cube_id = model.body("cube1").id
space_id = model.body("space").id
cube_pos = data.xpos[cube_id]
space_pos = data.xpos[space_id]

#Camera Rendering
h, w = 320, 480
renderer = mujoco.Renderer(model, height=h, width=w)
camera_name = "camera_head"

#Middle coordinate for camera screen
u = w // 2
v = h // 2


def smooth_move(current, target, speed=0.05):
    return current + speed * (target - current)

#Inverse Kinematics
def inverse_kinematics(model, data, hand_id, t_position, alpha = 0.3,q_nominal = False):

    mujoco.mj_kinematics(model,data)


    ee_position = data.xpos[hand_id].copy()
    ee_rotation = data.xmat[hand_id].reshape(3,3).copy()
 
    t_rotation = np.array([
        [1,0,0],
        [0,-1,0],
        [0,0,-1]
    ])

    pos_err = t_position - ee_position
    rot_err = t_rotation @ ee_rotation.T

    rot_vec = 0.5 * np.array([
    rot_err[2,1] - rot_err[1,2],
    rot_err[0,2] - rot_err[2,0],
    rot_err[1,0] - rot_err[0,1]
    ])

    err = np.concatenate([pos_err, rot_vec])
    
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    point = data.xpos[hand_id]
    mujoco.mj_jac(model,data,jac_pos, jac_rot, point,hand_id)
    J = np.vstack([jac_pos, jac_rot])

    JJt = J @ J.T
    dq = J.T @ np.linalg.solve(JJt + 0.1 * np.eye(6), err)

    q_current = data.qpos[:7].copy()
    q_target = q_current + alpha * dq[:7]

    data.ctrl[arm_actuator_ids] = q_target


def calculate_in_local(model, data, camera_name, cube_id):
    mujoco.mj_forward(model, data)
    camera_id = model.camera(camera_name).id
    world_cam = data.cam_xpos[camera_id]
    world_obj = data.xpos[cube_id]

    camera_rot = data.cam_xmat[camera_id].reshape(3,3)

    local_distance = camera_rot.T @ (world_obj - world_cam)

    return local_distance

def objects_in_fov(local):
    x, y, z = local
    if z >= 0:
        return False

    zz = -z

    return (abs(x / zz) <= 0.866) and (abs(y / zz) <= 0.577)


def project_local_to_pixel(local, w, h, fovy_deg):
    x, y, z = local
    if z >= 0:
        return None

    fovy = math.radians(fovy_deg)
    aspect = w / h
    fovx = 2.0 * math.atan(math.tan(fovy/2.0) * aspect)

    fx = (w/2.0) / math.tan(fovx/2.0)
    fy = (h/2.0) / math.tan(fovy/2.0)
    cx, cy = w/2.0, h/2.0

    u = cx + fx * (x / -z)
    v = cy + fy * (y / -z)
    return int(u), int(v)




with mujoco.viewer.launch_passive(model, data) as viewer:

    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    distance = depth[v, u] 

    state = 'wait'
    start_time = data.time
    t_position = data.xpos[hand_id].copy()
    goal_position = t_position.copy()


    mujoco.mj_forward(model, data)

    while viewer.is_running():


        current_time = data.time - start_time
        cube_pos = data.xpos[cube_id].copy()
        cam_id = model.camera(camera_name).id
        fovy_deg = float(model.cam_fovy[cam_id])

        local_value = calculate_in_local(model, data, camera_name, cube_id)
        

        if objects_in_fov(local_value):

        
            if state =='wait':
                goal_position = data.xpos[hand_id].copy()
                if current_time > 2.0:
                    state = 'start'

            elif state == 'start':
                goal_position = cube_pos + np.array([0, 0, 0.5])
                if current_time > 4.0:
                    state = 'above'

            elif state == 'above':
                data.ctrl[gripper_id] = 255
                if current_time > 6.0:
                    state = 'opened'
            
            elif state == 'opened':
                goal_position = cube_pos + np.array([-0.18,0,0.24])
                if current_time > 10.0:
                    state = 'down'

            elif state == 'down':
                data.ctrl[gripper_id] = 0
                if current_time > 12.0:
                    state = 'up'
            
            elif state =='up':
                goal_position = cube_pos + np.array([-0.2,0,0.6])
                if current_time > 14.0:
                    state = 'move'
            
            elif state == 'move':
                goal_position = space_pos + np.array([0,0,0.24])
                if current_time > 16.0:
                    state = 'finish'
            elif state == 'finish':
                data.ctrl[gripper_id] = 255
                goal_position = cube_pos + np.array([0, 0, 0.5])


        t_position = smooth_move(t_position, goal_position, speed=0.08)
        

        inverse_kinematics(model, data, hand_id, t_position, alpha=0.3)

        mujoco.mj_step(model, data)

        viewer.sync()

        
        renderer.update_scene(data,camera=camera_name)
        img = renderer.render()
        rgb = renderer.render()

        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()
        distance = depth[v, u] 
        
        calculate_in_local(model, data, camera_name,cube_id)



        cv2.imshow("Sub Camera", img[:, :, ::-1])

        
        if cv2.waitKey(1) == 27:
            break

        time.sleep(0.001)


cv2.destroyAllWindows()
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
space_id = model.body("target_space").id
start_pos_id = model.body("starting_space").id

start_pos = data.xpos[start_pos_id]
cube_pos = data.xpos[cube_id]
target_space_pos = data.xpos[space_id].copy()

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
def inverse_kinematics(model, data, hand_id, t_position, alpha = 0.3):

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

    #pseudo inverse
    j_pse_inverse = J.T @ np.linalg.solve(JJt + 0.01 * np.eye(6), np.eye(6))

    dq_t = j_pse_inverse @ err

    q_nominal = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8])
    q_cur = data.qpos[:7].copy()

    N = np.eye(model.nv) - j_pse_inverse @ J

    k_null = 0.5
    dq0 = np.zeros(model.nv)
    dq0[:7] = (q_nominal - q_cur)

    dq = dq_t + k_null * (N @ dq0)

    q_target = q_cur + alpha * dq[:7]  
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
    #need to change the numbers
    return (abs(x / zz) <= 0.866) and (abs(y / zz) <= 0.577)

def reached(ee_pos, goal_pos, tol):
    return np.linalg.norm(ee_pos - goal_pos) < tol

def check_fov(model, data, camera_name, scene_cubes):
    valid_cubes = []

    for cube_id in scene_cubes:
        local_value = calculate_in_local(model, data, camera_name, cube_id)

        if objects_in_fov(local_value):
            valid_cubes.append(cube_id)

    return valid_cubes
    


def pick_and_place(model,data, space_id,cube_id, ee_pos, state, state_start_time, gripper_open = 255, gripper_close = 0, tol = 0.07):

    start_pos_id = model.body("starting_space").id

    start_pos = data.xpos[start_pos_id]
    
    target_cube_pos = data.xipos[cube_id].copy()
    above_cube_pos = target_cube_pos + np.array([0, 0, 0.5])
    close_cube_pos = target_cube_pos + np.array([0, -0.015, 0.08])

    lift_cube_pos = start_pos + np.array([0, 0, 0.6])

    target_space_pos = data.xpos[space_id].copy()
    above_target_pos = target_space_pos + np.array([0, 0, 0.5])
    close_target_pos = target_space_pos + np.array([0, 0, 0.09])

    next_state = state
    goal_position = ee_pos.copy()


    if state == "start":
        goal_position = lift_cube_pos
        if reached(ee_pos,lift_cube_pos, tol):
            if data.time - state_start_time > 2:
                next_state = "move"



    elif state == "move":
        goal_position = above_cube_pos

        if reached(ee_pos,above_cube_pos, tol):
            next_state = "open_gripper"
        
    elif state == "open_gripper":
        data.ctrl[gripper_id] = gripper_open
        next_state = "descend_to_cube"
    
    elif state == "descend_to_cube":
        goal_position = close_cube_pos

        if reached(ee_pos, close_cube_pos,tol=0.04):
            next_state = "close_gripper"
        
    elif state == "close_gripper":
        
        data.ctrl[gripper_id] = gripper_close
        
        if data.time - state_start_time > 0.8:
            next_state = "lift"

    elif state == "lift":
        goal_position = lift_cube_pos
        if reached(ee_pos,lift_cube_pos,tol):
            next_state = "move_to_target"
    
    elif state == "move_to_target":
        goal_position = above_target_pos
        if reached(ee_pos,above_target_pos,tol):
            next_state = "descend_to_target"
        
    elif state == "descend_to_target":
        goal_position = close_target_pos
        if reached(ee_pos,close_target_pos,tol):
            data.ctrl[gripper_id] = gripper_open
            next_state = "move_up"
        
    elif state == "move_up":
        goal_position = above_target_pos
        if reached(ee_pos,above_target_pos,tol):
            next_state = "default"
        
    elif state == "default":
        goal_position = start_pos + np.array([0,0,0.5])
        if reached(ee_pos,start_pos + np.array([0,0,0.5]),tol):
            next_state = "end"


    return next_state, goal_position



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


    #Get the cube data
    scene_cubes = []

    for i in range(model.nbody):
        name = model.body(i).name
        if name is not None and name.startswith("cube"):
            scene_cubes.append(i)

    
    next_state = state
    state_start_time = data.time
    target_cube_id = None        


    while viewer.is_running():

        current_time = data.time - start_time
        cube_pos = data.xpos[cube_id].copy()
        cam_id = model.camera(camera_name).id
        fovy_deg = float(model.cam_fovy[cam_id])
        target_space_pos = data.xpos[space_id].copy()

        #starting position
        goal_position = start_pos + np.array([0,0,0.5])

        #filter valid cubes in fov
        valid_cubes = []

        for cube_id in scene_cubes:
            local_value = calculate_in_local(model, data, camera_name, cube_id)

            if objects_in_fov(local_value):
                valid_cubes.append(cube_id)

        if state == "wait":
            if len(valid_cubes) > 0:
                target_cube_id = valid_cubes[0]
                next_state = "start"

        elif state != "end":
            if target_cube_id is None:
                next_state = "wait"

            else:
                next_state, goal_position = pick_and_place(model,data, cube_id = target_cube_id,space_id = space_id, ee_pos=data.xpos[hand_id].copy(),state=state, state_start_time = state_start_time)

        
        if next_state != state:
            state_start_time = data.time

        state = next_state


        t_position = smooth_move(t_position, goal_position, speed=0.08)
        
        inverse_kinematics(model, data, hand_id, t_position, alpha=0.2)

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

        SHOW_CAMERA = True
        if SHOW_CAMERA:
            cv2.imshow("Sub Camera", img[:, :, ::-1])

        
            if cv2.waitKey(1) == 27:
                break

        time.sleep(0.002)


cv2.destroyAllWindows()
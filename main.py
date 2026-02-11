import numpy as np
import mujoco
import mujoco.viewer
import cv2

from motions import pick_and_place, reached, smooth_move
from detection import calculate_in_local, objects_in_fov, cube_length_check
from inverse_kinematics import inverse_kinematics


model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

arm_actuator_names = [
    "actuator1", "actuator2", "actuator3",
    "actuator4", "actuator5", "actuator6", "actuator7"
]


arm_actuator_ids = np.array([model.actuator(name).id for name in arm_actuator_names])
gripper_id = model.actuator("actuator8").id

#define values
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

#Max length of 2 finger joint
jid1 =  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
jid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")

r1 = model.jnt_range[jid1]
r2 = model.jnt_range[jid2]

finger1_max = r1[1] - r1[0]
finger2_max = r2[1] - r2[0]

#Max Length of Gripper
gripper_max_open  = finger1_max + finger2_max

#default gripper matrix
d_rotation = np.array([
    [1,0,0],
    [0,-1,0],
    [0,0,-1]
])

#Z-axis 90 rotation matrix
c, s = np.cos(np.pi/2), np.sin(np.pi/2)
z_90_rotation = np.array([
        [c,-s,0],
        [s,c,0],
        [0,0,1]
])

rotated = d_rotation @ z_90_rotation


with mujoco.viewer.launch_passive(model, data) as viewer:

    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    distance = depth[v, u] 

# Start timer
    state = 'wait'
    start_time = data.time
    t_position = data.xpos[hand_id].copy()
    goal_position = t_position.copy()

# Foward Kinematics
    mujoco.mj_forward(model, data)

# Get all cubes from MuJoCo
    scene_cubes = []
    # Put in in list
    for i in range(model.nbody):
        name = model.body(i).name
        if name is not None and name.startswith("cube"):
            scene_cubes.append(i)

    
    next_state = state
    state_start_time = data.time
    

# filter cubes that are in valid space
    valid_cubes = []     
    target_cube_id = None  
    exceeds_length = False


    while viewer.is_running():

        current_time = data.time - start_time
        cube_pos = data.xpos[cube_id].copy()
        cam_id = model.camera(camera_name).id
        fovy_deg = float(model.cam_fovy[cam_id])
        target_space_pos = data.xpos[space_id].copy()

        #starting position
        mujoco.mj_forward(model, data)

        #start position
        start_pos = data.xpos[start_pos_id].copy()
        #gripper position
        ee_pos = data.xpos[hand_id].copy()
        #default position for before pick up
        default_position = start_pos + np.array([0,0,0.5])
        at_default_position = reached(ee_pos, default_position, tol=0.05)

        #long rectangular box (Y-axis)
        exceeds_length = False

        #if state is "wait" it is ready to pick up the cube if it is on valid space
        if state =="wait":

            #move gripper to default postion (centre of limited space)
            goal_position = default_position
            
            
            if at_default_position:
                    cid = scene_cubes[0]
                    local = calculate_in_local(model, data, camera_name, cid)
                    
                    for cube_id in scene_cubes:
                        local_value = calculate_in_local(model, data, camera_name, cube_id)

                        if objects_in_fov(model,local_value,camera_name=camera_name,height=h,width=w):
                            valid_cubes.append(cube_id)


            if len(valid_cubes) > 0:
                target_cube_id = valid_cubes.pop(0)
                next_state = "start"

                cube_length_check

        elif state == "end":
            target_cube_id = None
            exceeds_length = False
            if len(valid_cubes) > 0:
                target_cube_id = valid_cubes.pop(0)
                next_state = "start"
            else:
                next_state = "wait"

    
        else:    
            next_state, goal_position = pick_and_place(model,data, gripper_id = gripper_id, cube_id = target_cube_id, space_id = space_id, ee_pos=data.xpos[hand_id].copy(),state=state, state_start_time = state_start_time)
            
        
        if next_state != state:
            state_start_time = data.time

        state = next_state


        t_position = smooth_move(t_position, goal_position, speed=0.1)
        

        if exceeds_length == False:
            t_rotation = d_rotation
            inverse_kinematics(model,data, hand_id, t_position, t_rotation, arm_actuator_ids,exceeds_length)
        else:
            t_rotation = rotated
            inverse_kinematics(model,data, hand_id, t_position, t_rotation, arm_actuator_ids,exceeds_length)

        mujoco.mj_step(model, data)

        viewer.sync()

        renderer.update_scene(data,camera=camera_name)
        
        #img = renderer.render()

        renderer.enable_depth_rendering()
        depth = renderer.render()
                
        renderer.disable_depth_rendering()
        distance = depth[v, u] 
        
        calculate_in_local(model, data, camera_name,cube_id)

        #cv2.imshow("Sub Camera", img[:, :, ::-1])

        if cv2.waitKey(1) == 27:
            break


cv2.destroyAllWindows()
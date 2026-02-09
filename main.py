import numpy as np
import mujoco
import mujoco.viewer
import cv2

from motions import pick_and_place, reached, smooth_move
from detection import calculate_in_local, objects_in_fov
from inverse_kinematics import inverse_kinematics


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

jid1 =  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
jid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")

r1 = model.jnt_range[jid1]
r2 = model.jnt_range[jid2]

finger1_max = r1[1] - r1[0]
finger2_max = r2[1] - r2[0]

gripper_max_open  = finger1_max + finger2_max


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

    scene_cubes = []
    for i in range(model.nbody):
        name = model.body(i).name
        if name is not None and name.startswith("cube"):
            scene_cubes.append(i)

    
    next_state = state
    state_start_time = data.time
    

    valid_cubes = []     
    target_cube_id = None  
    


    while viewer.is_running():

        current_time = data.time - start_time
        cube_pos = data.xpos[cube_id].copy()
        cam_id = model.camera(camera_name).id
        fovy_deg = float(model.cam_fovy[cam_id])
        target_space_pos = data.xpos[space_id].copy()

        #starting position
        mujoco.mj_forward(model, data)
        start_pos = data.xpos[start_pos_id].copy()
        ee_pos = data.xpos[hand_id].copy()
        default_position = start_pos + np.array([0,0,0.5])
        at_default_position = reached(ee_pos, default_position, tol=0.05)
        exceeds_length = False


        if state =="wait":
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
                geom_id = model.body_geomadr[target_cube_id]
                box_size = model.geom_size[geom_id]
                if box_size[1] * 2 > gripper_max_open:
                    exceeds_length = True
                else:
                    next_state = "start"

        elif state == "end":
            target_cube_id = None

            if len(valid_cubes) > 0:
                target_cube_id = valid_cubes.pop(0)
                next_state = "start"
            else:
                next_state = "wait"

        else:
            next_state, goal_position = pick_and_place(model,data, gripper_id = gripper_id,cube_id = target_cube_id,space_id = space_id, ee_pos=data.xpos[hand_id].copy(),state=state, state_start_time = state_start_time)

        
        if next_state != state:
            state_start_time = data.time

        state = next_state


        t_position = smooth_move(t_position, goal_position, speed=0.05)
        
        inverse_kinematics(model, data,hand_id=hand_id, arm_actuator_ids=arm_actuator_ids, exceeds_length = exceeds_length, t_position=t_position, alpha=0.3)

        mujoco.mj_step(model, data)

        viewer.sync()

        
        renderer.update_scene(data,camera=camera_name)
        
        img = renderer.render()

        renderer.enable_depth_rendering()
        depth = renderer.render()
                
        renderer.disable_depth_rendering()
        distance = depth[v, u] 
        
        calculate_in_local(model, data, camera_name,cube_id)

        cv2.imshow("Sub Camera", img[:, :, ::-1])

        if cv2.waitKey(1) == 27:
            break

        


cv2.destroyAllWindows()
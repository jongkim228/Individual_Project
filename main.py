import numpy as np
import mujoco
import mujoco.viewer
import cv2
import csv
import math

from motions import pick_and_place, reached, smooth_move
from detection import calculate_in_local, objects_in_fov, cube_length_check
from inverse_kinematics import inverse_kinematics
from packing import box_packing
from init import *

with mujoco.viewer.launch_passive(model, data) as viewer:

    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    distance = depth[v, u] 

# Start timer
    state = 'wait'
    start_time = data.time
    t_position = data.site_xpos[gripper_site_id].copy()
    goal_position = t_position

# Foward Kinematics
    mujoco.mj_forward(model, data)


# Get all cubes from MuJoCo
    scene_boxes = []
    # Put in in list
    for i in range(model.nbody):
        name = model.body(i).name
        if name is not None and name.startswith("cube"):
            scene_boxes.append(i)
  
    next_state = state
    state_start_time = data.time
    

# filter cubes that are in valid space
    valid_boxes = []     
    target_box_id = None  
    exceeds_length = None
    target_pack_pos = None
    packing_result = []
    areas = []
    sorted_boxes = []

    initialized = False
    t_rotation = d_rotation

    while viewer.is_running():

        #Timer
        current_time = data.time - start_time
        cam_id = model.camera(camera_name).id
        fovy_deg = float(model.cam_fovy[cam_id])
        target_space_pos = data.xpos[space_id].copy()

        #starting position
        mujoco.mj_forward(model, data)

        #start position
        start_pos = data.xpos[start_pos_id].copy()
        #gripper position
        ee_pos = data.site_xpos[gripper_site_id].copy()
        #default position for before pick up
        default_position = start_pos + np.array([0,0,0.5])
        at_default_position = reached(ee_pos, default_position, tol=0.05)

        gripper_pos = data.site_xpos[gripper_site_id].copy()

        

        #if state is "wait" it is ready to pick up the cube if it is on valid space
        if state == "wait":
            #move gripper to default postion (centre of limited space)
            goal_position = default_position

            if at_default_position and not initialized:
                for scene_box in scene_boxes:
                    box_geom_id = model.body_geomadr[scene_box]
                    local_value = calculate_in_local(model, data, camera_name, box_geom_id)

                    if objects_in_fov(model,local_value,camera_name=camera_name, height=h,width=w):
        
                        valid_boxes.append(scene_box)
                        num_box += 1


                print(f"{num_box} boxes have detected")

                for i in valid_boxes:
                    geom_id = model.body_geomadr[i]
                    size = model.geom_size[geom_id].copy()
                                            
                    areas.append(max(size) * 100)

                sorted_boxes = sorted(valid_boxes,key=lambda i: dict(zip(valid_boxes, areas))[i],reverse=True)
                initialized = True


                #Box on valid space
                if len(sorted_boxes) > 0:
                    packing_result = box_packing(data, model, sorted_boxes)

                    target_pack_pos = packing_result.pop(0)
                    target_box_id = sorted_boxes.pop(0)

                    valid_box_geom = model.body_geomadr[target_box_id]

                    #check the box length is over the gripper open range
                    exceeds_length = cube_length_check(model,valid_box_geom,gripper_max_open)


                    next_state = "start"
                    

        elif state == "end":
            if len(sorted_boxes) > 0:
                target_box_id = sorted_boxes.pop(0)
                target_pack_pos = packing_result.pop(0)

                end_box_geom = model.body_geomadr[target_box_id]
                exceeds_length = cube_length_check(model, end_box_geom, gripper_max_open)


                next_state = "start"

            else:
                target_box_id = None
                exceeds_length = None
                next_state = "wait"

            state_start_time = data.time

    
        else:
            if target_box_id is not None:
                valid_geom = model.body_geomadr[target_box_id]
                target_box = data.geom_xpos[valid_geom].copy()

                if exceeds_length == "long":
                    t_rotation = long_rotated
                elif exceeds_length == "tall":
                    t_rotation = tall_rotated
                else:
                    t_rotation = d_rotation


                next_state, goal_position = pick_and_place(model, data, gripper_id, target_box, ee_pos, state, state_start_time, pack_pos=target_pack_pos, rotation=exceeds_length)

        if next_state != state:
            state_start_time = data.time
            if next_state == "start":
                print("--------------------")
                print("Started Moving Boxes")
            elif next_state == "open_gripper":
                print("Open Gripper")
            elif next_state == "descend_to_cube":
                print("Approach to the box")
            elif next_state == "close_gripper":
                print("Close Gripper")
            elif next_state == "lift":
                print("Lift the box")
            elif next_state == "move":
                print("Move to target position")
            elif next_state == "drop":
                print("Approach to drop")
            elif next_state == "release_gripper":
                print("Release gripper")
            elif next_state == "move_to_default":
                print("Move to default")
            elif next_state == "move_to_start":
                print("Move to start")
            elif next_state == "end":
                print("Task End")

        state = next_state

        if state == "close_gripper":
            pass
        else:
            t_position = smooth_move(t_position, goal_position, speed=0.7)
        

        for _ in range(10): 
            if state != "close_gripper":
                inverse_kinematics(model, data, gripper_site_id, t_position, t_rotation, arm_actuator_ids, exceeds_length, alpha=0.3)
                    
                mujoco.mj_kinematics(model, data)

        mujoco.mj_step(model, data)


        viewer.sync()

        #renderer.update_scene(data,camera=camera_name)
        
        #img = renderer.render()

        renderer.enable_depth_rendering()
        depth = renderer.render()
                
        renderer.disable_depth_rendering()
        distance = depth[v, u] 
        

        #cv2.imshow("Sub Camera", img[:, :, ::-1])

        if cv2.waitKey(1) == 27:
            break


cv2.destroyAllWindows()
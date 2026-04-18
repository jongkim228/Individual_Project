import numpy as np
import mujoco
import mujoco.viewer
import cv2
import csv
import math

from motions import pick_and_place, reached, smooth_move
from detection import calculate_in_local, objects_in_fov, cube_length_check
from inverse_kinematics import inverse_kinematics
from packing import box_solution
from init import *
from collision import collision_check


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
    t_rotation = d_rotation.copy()

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
                if default_q_nominal is None:
                    default_q_nominal = data.qpos[:7].copy()
                for scene_box in scene_boxes:
                    box_geom_id = model.body_geomadr[scene_box]
                    local_value = calculate_in_local(model, data, camera_name, box_geom_id)

                    if objects_in_fov(model,local_value,camera_name=camera_name, height=h,width=w):
        
                        valid_boxes.append(scene_box)
                        num_box += 1

                print(f"{num_box} boxes have detected")


                sorted_boxes = valid_boxes.copy()
                initialized = True

                #Box on valid space
                if len(sorted_boxes) > 0:
                    packing_result = box_solution(data, model, sorted_boxes,placed_boxes)
                    if len(packing_result) == 0:
                        print("No Packing Solution Found")
                        break
                    all_solutions = packing_result.copy()
                    target_box_solution = packing_result.pop(0)
                    target_box_id = sorted_boxes.pop(0)
                    fixed_box_xy = data.xpos[target_box_id]

                    print(target_box_id)

                    valid_box_geom = model.body_geomadr[target_box_id]
                    #check the box length is over the gripper open range
                    grip_dir = cube_length_check(model,valid_box_geom,gripper_max_open)
                    placed_solutions = all_solutions[:len(placed_boxes)]
                    
                    next_state = "start"
                    

        elif state == "end":
            #After putting box on the target space
            #put the box id in to placed_boxes list
            placed_boxes.append(target_box_id)
            geom_id = model.body_geomadr[target_box_id]
            actual_pos = data.geom_xpos[geom_id].copy()
            placement_log.append((target_box_id, target_box_solution, actual_pos))    

            # if there are more boxes
            if len(sorted_boxes) > 0:
                #Get reamining box details
                target_box_id = sorted_boxes.pop(0)
                target_box_solution = packing_result.pop(0)
                end_box_geom = model.body_geomadr[target_box_id]

                #Check the collison and decide how to grab a box
                grip_dir = cube_length_check(model, end_box_geom, gripper_max_open)
                placed_solutions = all_solutions[:len(placed_boxes)] 

                next_state = "start"

            else:
                target_box_id = None
                grip_dir = None
                next_state = "wait"

            state_start_time = data.time

    
        else:
            if target_box_id is not None:
                valid_geom = model.body_geomadr[target_box_id]
                target_box = data.geom_xpos[valid_geom].copy()

                if state in ["start", "open_gripper", "move_to_above_cube", "descend_to_cube"]:
                    fixed_box_xy = data.xpos[target_box_id][:2].copy()

                before_rotate_states = ["start", "open_gripper", "move_to_above_cube", 
                       "descend_to_cube", "close_gripper", "lift"]
                if state in before_rotate_states:
                    if grip_dir == "x_axis":
                        t_rotation = long_rotated
                    else:
                        t_rotation = d_rotation

                next_state, goal_position, captured_q_nominal, t_rotation, pack_rotation, fixed_box_xy = pick_and_place(
                fixed_box_xy,
                model, data, gripper_id, target_box, target_box_id, ee_pos,
                state, state_start_time,
                d_rotation,
                pack_pos=target_box_solution,
                grip_dir=grip_dir,
                gripper_site_id=gripper_site_id,
                t_rotation=t_rotation,
                placed_boxes=placed_boxes,
                placed_solutions=placed_solutions,
                target_box_solution=target_box_solution,
                default_q_nominal = default_q_nominal
            )

                if captured_q_nominal is not None:
                    saved_q_nominal = captured_q_nominal

        if next_state != state:
            state_start_time = data.time
            if next_state == "start":
                print("==============================")
                print("Started Moving Boxes")
            elif next_state == "open_gripper":
                print("Open Gripper")
            elif next_state == "move_to_above_cube":
                print("Move to above cube")
            elif next_state == "descend_to_cube":
                print("Approach to the box")
            elif next_state == "close_gripper":
                print("Close Gripper")
            elif next_state == "lift":
                print("Lift the box")
            elif next_state == "rotate_check":
                if pack_rotation == 2:
                    p_rotaion = "Vertically"
                    print(f"Turning The Box {p_rotaion}")
                elif pack_rotation == 0:
                    print("Not Using Rotations")
                else:
                    p_rotaion = "Sideways"
                    print(f"Turning The Box {p_rotaion}")
            elif next_state == "move":
                print("Move To Target Position")
            elif next_state == "collision_check_state":
                print("Checking Collision")
            elif next_state == "rotate_gripper":
                print("Rotating Gripper")
            elif next_state == "move_to_place":
                print("Move To Place Position")
            elif next_state == "place":
                print("Placing Box")
            elif next_state == "move_to_drop":
                print("Move To Drop Position")
            elif next_state == "release_gripper":
                print("Release Gripper")
            elif next_state == "move_to_default":
                print("Move To Default")
            elif next_state == "move_to_start":
                print("Move to start")
            elif next_state == "end":
                print("Task End")

        state = next_state

        if state in ["descend_to_cube", "lift_up", "drop", "place"]:
            t_position = smooth_move(t_position, goal_position, speed=0.1)
        elif state in ["move"]:
            t_position = smooth_move(t_position, goal_position, speed=0.05)
        else:
            t_position = smooth_move(t_position, goal_position, speed=0.15)
        


        for _ in range(10): 
            if state != "close_gripper":
                ik_params = params.get(state, {"alpha": 0.3, "k_null": 0.05, "damping": 0.05})
                if state in ["descend_to_cube", "lift"]:
                    target_q_nom = saved_q_nominal
                else:
                    target_q_nom = None
                
                inverse_kinematics(model, data, gripper_site_id, t_position, t_rotation, arm_actuator_ids, custom_q_nominal=target_q_nom, **ik_params)
                mujoco.mj_forward(model, data)

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

print("\n========PLACEMENT SUMMARY========\n")
with open("placement_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["box", "intended_x", "intended_y", "intended_z",
        "actual_x", "actual_y", "actual_z", "error_dist"])
    writer.writeheader()
    for i, (box_id, solution, actual_pos) in enumerate(placement_log):
        intended_pos = np.array([solution["x"], solution["y"], solution["z"]])
        error_dist = np.linalg.norm(actual_pos - intended_pos)
        box_name = model.body(box_id).name
        print(f"[{i+1}] {box_name} | error: {error_dist:.4f}m")
        errors.append(error_dist)
        writer.writerow({
            "box": box_name,
            "intended_x": round(intended_pos[0], 3),
            "intended_y": round(intended_pos[1], 3),
            "intended_z": round(intended_pos[2], 3),
            "actual_x": round(actual_pos[0], 3),
            "actual_y": round(actual_pos[1], 3),
            "actual_z": round(actual_pos[2], 3),
            "error_dist": round(error_dist, 3)
        })

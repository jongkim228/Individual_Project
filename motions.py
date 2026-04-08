import numpy as np
import mujoco
from init import *
from collision import collision_check

log_file = open("error_log.txt", "w")

def smooth_move(current, target, speed=0.03):
    diff = target - current
    if np.linalg.norm(diff) < 0.02:
        return target.copy() 
    return current + speed * (diff)


def reached(ee_pos, goal_pos, tol):
    return np.linalg.norm(ee_pos - goal_pos) < tol

    
def pick_and_place(    
    fixed_box_xy,
    model,
    data,
    gripper_id,
    box_pos,
    box_id,
    ee_pos,
    state,
    state_start_time,
    d_rotation,
    grip_dir,
    gripper_site_id=None, 
    t_rotation=None,        
    pack_pos=None,
    placed_boxes = None,
    placed_solutions=None,
    target_box_solution = None,
    default_q_nominal = None,
    tol=0.04):
    
    captured_q_nominal = None
    pack_rotation = None
    rotated = None

    box_name = model.body(box_id).name
    geom_id = model.geom(box_name).id
    box_size = model.geom_size[geom_id]
    z_value = box_size[2]


    # Get Space ID 
    start_pos_id = model.body("starting_space").id
    target_pos_id = model.body("target_space").id

    # Get space position coordinates
    start_space = data.xpos[start_pos_id]
    target_space = data.xpos[target_pos_id]

    # get target box coordinate
    target_box_pos = box_pos.copy()

    # coordinate for above target box
    above = np.array([0, 0, 0.5])
    above_box_pos = target_box_pos + above


    close = np.array([0, 0, - z_value +0.01])


    drop = np.array([0,0,0.01])
    pick_pos = target_box_pos + close


    # gripper id and control range
    gripper_id = model.actuator("actuator8").id
    gripper_range = model.actuator(gripper_id).ctrlrange
    
    # max gripper close and grippe open
    gripper_close = gripper_range[0]
    gripper_open = gripper_range[1]

    #default postion when gripper picked up the box
    lift_pos = start_space + np.array([0, 0, 0.5])

    #Before drop
    above_target = target_space + above

    next_state = state
    goal_position = ee_pos.copy()
    current = ee_pos.copy()

    captured_q_nominal = None
    rotate_q_nominal = None


    if pack_pos is not None:
        place_pos = np.array([pack_pos["x"], pack_pos["y"], pack_pos["z"] - 0.01])

    # Start the task
    if state == "start":
        # move gripper to above the cube
        goal_position = above_box_pos
        # change the state to open the gripper
        if reached(current,goal_position, tol = 0.05):
            next_state = "open_gripper"
        
    #open gripper    
    elif state == "open_gripper":
        data.ctrl[gripper_id] = gripper_open

        if data.time - state_start_time > 0.2:
            captured_q_nominal = default_q_nominal
            next_state = "move_to_above_cube"

    elif state == "move_to_above_cube":
        goal_position =  np.array([target_box_pos[0],
        target_box_pos[1], 
        0.5 ])
        
        if reached(current, goal_position, tol=0.03):
            if data.time - state_start_time > 0.3:
                next_state = "descend_to_cube"


    elif state == "descend_to_cube":
        current = ee_pos.copy()
        goal_position =  np.array([target_box_pos[0],
        target_box_pos[1], 
        pick_pos[2] ])
        
        if reached(current, goal_position, tol=0.02):
            if data.time - state_start_time > 0.3:
                next_state = "close_gripper"


    elif state == "close_gripper":
        goal_position = current
        fixed_box_xy = current[:2].copy()
        data.ctrl[gripper_id] = gripper_close
        
        if data.time - state_start_time > 0.3:
            next_state = "lift"

    elif state == "lift":
        data.ctrl[gripper_id] = gripper_close
        goal_position = np.array([fixed_box_xy[0],fixed_box_xy[1],0.5])

        if reached(current,goal_position,tol = 0.07):
            print(rotate_q_nominal)
            next_state = "rotate_check"

    elif state == "rotate_check":
        data.ctrl[gripper_id] = gripper_close
        goal_position = np.array([start_space[0], start_space[1], 0.6])


        pack_rotation = pack_pos["rotation"]
        c, s = np.cos(np.pi/2), np.sin(np.pi/2)
        
        if grip_dir == "x_axis":
            z_90 = np.array([[c,-s,0], [s,c,0], [0,0,1]])
            rotated = True
            base_rotation = d_rotation @ z_90
        else:
            base_rotation = d_rotation
        print(pack_rotation)

        if pack_rotation == 1:
            R = np.array([[0, 1, 0],[-1,  0, 0],[0,  0, 1]])

        elif pack_rotation == 2 or pack_rotation == 4 or pack_rotation == 5:
            print(rotated)
            if rotated:
                R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                
            else:
                R_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
                R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                R = R_y @ R_x

        elif pack_rotation == 3:
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        else:
            R = np.eye(3)
        
        t_rotation = base_rotation @ R
        current_rotation = data.site_xmat[gripper_site_id].reshape(3, 3)

        rot_err = np.linalg.norm(t_rotation - current_rotation, 'fro')

        if rot_err < 0.07:
            next_state = "move"

    elif state == "move":
        goal_position = np.array([target_space[0],target_space[1],0.5])
        if reached(current,goal_position,tol = 0.05):
            next_state = "collision_check_state"

    elif state == "collision_check_state":
        collision_result, grip_dir = collision_check(box_id,grip_dir,placed_boxes,target_box_solution,placed_solutions)
        print(f"[ROTATION CHECK]{grip_dir}")
        print(f"COLLISION CHECK]{collision_result}")
        if collision_result == "safe":
            next_state = "move_to_place"
        elif collision_result == "rotate":
            print(f"[ROTATION CHECK]{grip_dir}")
            print("rotate!")
            next_state = "rotate_gripper"
        else: #drop
            next_state = "move_to_drop"

    elif state == "rotate_gripper":
            c, s = np.cos(np.pi/2), np.sin(np.pi/2)
            z_90 = np.array([
                [c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]
            ])

            if t_rotation is None or np.allclose(t_rotation, d_rotation, atol=0.2):
                current = data.site_xmat[gripper_site_id].reshape(3, 3).copy()
                t_rotation = current @ z_90

            current_rotation = data.site_xmat[gripper_site_id].reshape(3, 3)
            rot_err = np.linalg.norm(t_rotation - current_rotation, 'fro')
            
            if rot_err < 0.05:
                grip_dir = "y_axis" if grip_dir == "x_axis" else "x_axis"   
                next_state = "move_to_place"


    #Move to above target coordinate for vertical move
    elif state == "move_to_drop":
        goal_position = np.array([place_pos[0],place_pos[1],0.3])
        if reached(current,goal_position,tol = 0.05):
            next_state = "release_gripper"

    elif state == "move_to_place":
        goal_position = np.array([place_pos[0],place_pos[1],0.3])
        if reached(current,goal_position,tol = 0.05):
            next_state = "place"

    elif state == "place":
        goal_position = place_pos
        if reached(current, goal_position, tol=0.03):
            next_state = "release_gripper"

    elif state == "release_gripper":
        data.ctrl[gripper_id] = gripper_open
        if data.time - state_start_time > 0.3:
            next_state = "move_to_default"

    elif state == "move_to_default":
            goal_position = np.array([place_pos[0], place_pos[1], 0.5])
            if reached(current,goal_position,tol):
                next_state = "move_to_start"

    elif state == "move_to_start":
        goal_position = lift_pos
        if reached(current,goal_position,tol):
            next_state = "end"

    return next_state, goal_position, captured_q_nominal, t_rotation, pack_rotation, fixed_box_xy
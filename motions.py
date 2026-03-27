import numpy as np
import mujoco

def smooth_move(current, target, speed=0.1):
    return current + speed * (target - current)


def reached(ee_pos, goal_pos, tol):
    return np.linalg.norm(ee_pos - goal_pos) < tol

    
def pick_and_place(    
    model,
    data,
    gripper_id,
    box_pos,
    box_id,
    ee_pos,
    state,
    state_start_time,
    rotation,
    pack_pos = None,
    tol=0.04):
    

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

    # coordinate for box pick up
    if rotation == "long":
        close = np.array([0, 0, -z_value])

    else:
        close = np.array([0,-0.02, -z_value])


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

    if pack_pos is not None:
        drop_pos = np.array([pack_pos["x"], pack_pos["y"], pack_pos["z"] - 0.01])
        
    else:
        drop_pos = target_space + drop

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
        # wait until it opens the gripper
        if data.time - state_start_time > 0.4:
            next_state = "descend_to_cube"
    
    elif state == "descend_to_cube":
        current = ee_pos.copy()
        goal_position = pick_pos
        
        if reached(current, goal_position, tol=0.05):
            if data.time - state_start_time > 0.3:
                next_state = "close_gripper"


    elif state == "close_gripper":
        goal_position = current
        data.ctrl[gripper_id] = 0
        
        if data.time - state_start_time > 1.2:
            next_state = "lift"

    elif state == "lift":
        data.ctrl[gripper_id] = gripper_close
        goal_position = lift_pos
        if reached(current,goal_position,tol):
            next_state = "move"

    elif state == "move":

        goal_position = above_target

        if reached(current,goal_position,tol = 0.05):
            next_state = "drop"

    elif state == "drop":
        goal_position = drop_pos
        if reached(current, drop_pos, tol=0.08):
            next_state = "release_gripper"

    elif state == "release_gripper":
        data.ctrl[gripper_id] = gripper_open
        next_state = "move_to_default"

    elif state == "move_to_default":
        if data.time - state_start_time > 0.3:
            goal_position = above_target

            if reached(current,goal_position,tol):
                next_state = "move_to_start"

    elif state == "move_to_start":
        goal_position = lift_pos
        if reached(current,goal_position,tol):
            next_state = "end"


    return next_state, goal_position
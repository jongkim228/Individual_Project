import numpy as np
import mujoco

def smooth_move(current, target, speed=0.1):
    return current + speed * (target - current)


def reached(ee_pos, goal_pos, tol):
    return np.linalg.norm(ee_pos - goal_pos) < tol

    
def pick_and_place(model,data, exceeds_length,gripper_id, space_id,cube_id, ee_pos, state, state_start_time, gripper_open = 255, gripper_close = 0, tol = 0.07):

    start_pos_id = model.body("starting_space").id

    start_pos = data.xpos[start_pos_id]
    
    target_cube_pos = data.xpos[cube_id].copy()
    above_cube_pos = target_cube_pos + np.array([0, 0, 0.5])

    if exceeds_length:
        close_cube_pos = target_cube_pos + np.array([0.01, 0, 0.08])
    else:
        close_cube_pos = target_cube_pos + np.array([0.01, -0.015, 0.08])


    lift_cube_pos = start_pos + np.array([0, 0, 0.6])

    target_space_pos = data.xpos[space_id].copy()
    above_target_pos = target_space_pos + np.array([0, 0, 0.5])
    close_target_pos = target_space_pos + np.array([0, 0, 0.09])

    next_state = state
    goal_position = ee_pos.copy()

    # Start the task
    if state == "start":
        # move gripper to above the cube
        goal_position = above_cube_pos
        # change the state to open the gripper
        if reached(ee_pos,above_cube_pos, tol):
            next_state = "open_gripper"
        
    #open gripper    
    elif state == "open_gripper":
        data.ctrl[gripper_id] = gripper_open
        #wait until it opens the gripper and approach to the cube
        if data.time - state_start_time > 0.8:
            next_state = "descend_to_cube"
    
    elif state == "descend_to_cube":
        goal_position = close_cube_pos
        if reached(ee_pos, close_cube_pos,tol=0.04):
            next_state = "close_gripper"
        
    elif state == "close_gripper":
        
        data.ctrl[gripper_id] = gripper_close
        print("time:", data.time,
      "state:", state,
      "gripper ctrl:", data.ctrl[gripper_id])

        
        if data.time - state_start_time > 1:
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
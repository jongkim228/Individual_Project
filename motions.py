import numpy as np
import mujoco

def smooth_move(current, target, speed=0.1):
    return current + speed * (target - current)


def reached(ee_pos, goal_pos, tol):
    return np.linalg.norm(ee_pos - goal_pos) < tol

    
def pick_and_place(    model,
    data,
    exceeds_length,
    t_rotation,
    gripper_id,
    space_id,
    target_site_id,
    ee_pos,
    state,
    state_start_time,
    tol=0.07):

    start_pos_id = model.body("starting_space").id

    start_pos = data.xpos[start_pos_id]
    
    # get target box coordinate
    target_cube_pos = data.site_xpos[target_site_id].copy()

    # coordinate for above target box
    above = np.array([0, 0, 0.1])
    above_target_pos = target_cube_pos + above

    # coordinate for box pick up
    pick = np.array([0.3,0,0.03])
    pick_target_pos = target_cube_pos + pick

    # gripper id and control range
    gripper_id = model.actuator("actuator8").id
    gripper_range = model.actuator(gripper_id).ctrlrange
    
    # max gripper close and grippe open
    gripper_close = gripper_range[0]
    gripper_open = gripper_range[1]

    #default postion when gripper picked up the box
    lift_cube_pos = start_pos + np.array([0, 0, 0.6])

    next_state = state
    goal_position = ee_pos.copy()
    current = ee_pos.copy()

    # Start the task
    if state == "start":
        # move gripper to above the cube
        goal_position = above_target_pos
        # change the state to open the gripper
        if reached(current,goal_position, tol):
            next_state = "open_gripper"
        
    #open gripper    
    elif state == "open_gripper":
        data.ctrl[gripper_id] = gripper_open
        # wait until it opens the gripper
        if data.time - state_start_time > 0.8:
            next_state = "descend_to_cube"
    
    elif state == "descend_to_cube":
        current = ee_pos.copy()
        goal_position = pick_target_pos
        distance = np.linalg.norm(current - goal_position)
        diff = current - goal_position
        
        print(f"\n[DESCEND 상세]")
        print(f"  목표 (상자 중심): {goal_position}")
        print(f"  현재 (site): {current}")
        print(f"  차이:")
        print(f"    X: {diff[0]*100:+.2f}cm")
        print(f"    Y: {diff[1]*100:+.2f}cm")
        print(f"    Z: {diff[2]*100:+.2f}cm")
        print(f"  총 거리: {distance*100:.2f}cm")
        
        if reached(current, goal_position, tol=0.03):
            if data.time - state_start_time > 1.0:
                print(f"[DESCEND] Close로 전환! (최종 거리: {distance*100:.2f}cm)")
                next_state = "close_gripper"


    elif state == "close_gripper":
        goal_position = current
        data.ctrl[gripper_id] = gripper_close
        
        if data.time - state_start_time > 1:
            next_state = "lift"

    elif state == "lift":
        data.ctrl[gripper_id] = gripper_close
        goal_position = lift_cube_pos
        if reached(current,goal_position,tol):
            print("finish")



    return next_state, goal_position
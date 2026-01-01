import numpy as np
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

arm_actuator_names = [
    "actuator1", "actuator2", "actuator3",
    "actuator4", "actuator5", "actuator6", "actuator7"
]


arm_actuator_ids = np.array([model.actuator(name).id for name in arm_actuator_names])


actuator1 = model.actuator(arm_actuator_names[0]).id
actuator2 = model.actuator(arm_actuator_names[1]).id
actuator3 = model.actuator(arm_actuator_names[2]).id
actuator4 = model.actuator(arm_actuator_names[3]).id
actuator5 = model.actuator(arm_actuator_names[4]).id
actuator6 = model.actuator(arm_actuator_names[5]).id
gripper = model.actuator(arm_actuator_names[6]).id


hand_id = model.body("hand").id
hand_pos = data.xpos[hand_id]

cube_id = model.body("cube1").id
cube_pos = data.xpos[cube_id]

def ik(model, data, hand_id, t_position):

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
    jac = np.vstack([jac_pos, jac_rot])

    dq = np.linalg.pinv(jac) @ err
    data.qvel[:7] = dq[:7]


with mujoco.viewer.launch_passive(model, data) as viewer:

    state = 'wait'
    start_time = data.time

    t_position = data.xpos[hand_id].copy()

    while viewer.is_running():
        current_time = data.time - start_time
        ik(model, data, hand_id, t_position)
        
        if state =='wait':
            if current_time > 2.0:
                state = 'start'

        elif state == 'start':
            t_position = cube_pos + np.array([0, 0, 0.5])
            if current_time > 4.0:
                state = 'above'

        elif state == 'above':
            data.ctrl[7] = 255
            if current_time > 7.0:
                state = 'opened'

        elif state == 'opened':
            t_position = cube_pos + np.array([0.01, 0, 0.17])
            if current_time > 12.0:
                state = 'down'

        elif state == 'down':
            data.ctrl[7] = 0
            if current_time > 14.0:
                state = 'grab'
                
        elif state == 'grab':
            data.ctrl[7] = 0
            if current_time > 16.0:
                t_position = cube_pos + np.array([0, 0, 0.5])
                state = 'move'

        elif state == 'move':
            t_position =  np.array([0.5, 0.3, 0.5])
            

        mujoco.mj_step(model, data)
        viewer.sync()


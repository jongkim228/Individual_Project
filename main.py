import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

arm_actuator_names = [
    "actuator1", "actuator2", "actuator3",
    "actuator4", "actuator5", "actuator6", "actuator7"
]

arm_actuator_ids = np.array(
    [model.actuator(name).id for name in arm_actuator_names]
)

hand_id = model.body("hand").id
cube_id = model.body("cube1").id

act_id2 = model.actuator("actuator2").id
act_id3 = model.actuator("actuator3").id
act_id4 = model.actuator("actuator4").id
act_id5 = model.actuator("actuator5").id



def ik_lookat(model, data, hand_id, target_pos, cube_pos,
              kp_pos=1.5, kp_ori=1.5):

    mujoco.mj_forward(model, data)

    ee_pos = data.xpos[hand_id]
    pos_err = target_pos - ee_pos

    J_pos = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, J_pos, None, ee_pos, hand_id)

    R = data.xmat[hand_id].reshape(3, 3)
    ee_z = R @ np.array([0.0, 0.0, 1.0])

    target_dir = cube_pos - ee_pos
    target_dir /= np.linalg.norm(target_dir)

    ori_err = np.cross(ee_z, target_dir)

    J_ori = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, None, J_ori, ee_pos, hand_id)

    dq = kp_pos * (J_pos.T @ pos_err) + kp_ori * (J_ori.T @ ori_err)
    dq = np.clip(dq, -2.0, 2.0)

    return dq





with mujoco.viewer.launch_passive(model, data) as viewer:
    state = "Above"
    while viewer.is_running():
        cube_pos = data.xpos[cube_id].copy()
        hand_pos = data.xpos[hand_id].copy()

        

        if state == "Above":

            target_pos = cube_pos.copy()
            target_pos[2] += 0.5

            dq = ik_lookat(model, data, hand_id, target_pos, cube_pos)

            data.qvel[:] = dq
            
            if np.linalg.norm(hand_pos - target_pos) < 0.02:
                state = "Descend"

        elif state == "Descend":
            data.ctrl[act_id4] = -1


        mujoco.mj_step(model, data)
        viewer.sync()

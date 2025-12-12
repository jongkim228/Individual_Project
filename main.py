import numpy as np
import mujoco
import mujoco.viewer
import cv2
import glfw

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

arm_actuator_names = [
    "actuator1", "actuator2", "actuator3",
    "actuator4", "actuator5", "actuator6", "actuator7"
]


arm_actuator_ids = np.array([model.actuator(name).id for name in arm_actuator_names])

hand_id = model.body("hand").id
hand_pos = data.xpos[hand_id]
cube_id = model.body("cube1").id
cube_pos = data.xpos[cube_id]

def ik_lookat(model, data, hand_id, target_pos, cube_pos, kp_pos=1.5, kp_ori=1.5):
    mujoco.mj_forward(model, data)

    ee_pos = data.xpos[hand_id]
    pos_err = target_pos - ee_pos

    J_pos = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, J_pos, None, ee_pos, hand_id)

    R = data.xmat[hand_id].reshape(3,3)
    ee_z = R @ np.array([0.0, 0.0, 1.0])
    target_dir = cube_pos - ee_pos
    target_dir /= np.linalg.norm(target_dir)

    ori_err = np.cross(ee_z, target_dir)

    J_ori = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, None, J_ori, ee_pos, hand_id)

    dq = kp_pos * (J_pos.T @ pos_err) + kp_ori * (J_ori.T @ ori_err)
    data.qvel[:] = dq

    mujoco.mj_step(model, data)



with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for _ in range(600):

            cube_pos = data.xpos[cube_id].copy()

            target_pos = cube_pos.copy()
            target_pos[2] += 0.40

            ik_lookat(model, data, hand_id, target_pos, cube_pos)
            


            viewer.sync()

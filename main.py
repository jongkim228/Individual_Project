import numpy as np
import mujoco
import mujoco.viewer


model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)


arm_actuator_names = [
    "actuator1", "actuator2", "actuator3",
    "actuator4", "actuator5", "actuator6", "actuator7"
]
arm_actuator_ids = np.array([model.actuator(name).id for name in arm_actuator_names])

hand_id = model.body("hand").id
hand_pos = data.xpos[hand_id]
cube_id = model.body("cube1").id
cube_pos = data.xpos[cube_id]

def ik(model, data, hand_id, target_pos, kp=1.5):
    mujoco.mj_forward(model, data)
    ee_pos = data.xpos[hand_id]

    error = target_pos - ee_pos

    J = np.zeros((3, model.nv))
    point = ee_pos.reshape(3, 1)

    mujoco.mj_jac(model, data, J, None, point, hand_id)

    dq = kp * J.T @ error
    data.qvel[:] = dq



with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for _ in range(600):

            cube_pos = data.xpos[cube_id].copy()

            target_pos = cube_pos.copy()
            target_pos[2] += 0.10

            ik(model, data, hand_id, target_pos)

            mujoco.mj_step(model, data)

            viewer.sync()

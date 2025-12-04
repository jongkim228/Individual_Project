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


def move_forward_live():
    data.ctrl[:] = 0.0
    for _ in range(600):
        data.ctrl[arm_actuator_ids[1]] = 0.5
        data.ctrl[arm_actuator_ids[2]] = -0.5
        data.ctrl[arm_actuator_ids[3]] = 1
        data.ctrl[arm_actuator_ids[4]] = 0.5
        data.ctrl[arm_actuator_ids[4]] = 0.5

        mujoco.mj_step(model, data)
        viewer.sync()


with mujoco.viewer.launch_passive(model, data) as viewer:
    move_forward_live()


    # 끝난 뒤 카메라 유지
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

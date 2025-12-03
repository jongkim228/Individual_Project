import time
import math
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    'mujoco_menagerie/franka_fr3/scene.xml'
)

data = mujoco.MjData(model)

print("Actuator 개수:", model.nu)  # 확인용 출력
print("Actuator 타입:", model.actuator_trntype)
print("Actuator 제어 대상:", [model.actuator_trnid[i][0] for i in range(model.nu)])
print("초기 관절 위치:", data.qpos[:7])



with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        # 첫 번째 관절을 계속 좌우로 흔들기
        data.ctrl[0] = 0.7 * math.sin(2 * (time.time() - start))

        mujoco.mj_step(model, data)
        time.sleep(0.002)

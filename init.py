import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

arm_actuator_names = [
    "actuator1", "actuator2", "actuator3",
    "actuator4", "actuator5", "actuator6", "actuator7"
]


arm_actuator_ids = np.array([model.actuator(name).id for name in arm_actuator_names])
gripper_id = model.actuator("actuator8").id



# space
space_id = model.body("target_space").id
start_pos_id = model.body("starting_space").id
start_pos = data.xpos[start_pos_id].copy()
target_space_pos = data.xpos[space_id].copy()

# gripper site
gripper_site_id = model.site("gripper").id



#Camera Rendering
h, w = 320, 480
renderer = mujoco.Renderer(model, height=h, width=w)
camera_name = "camera_head"

#Middle coordinate for camera screen
u = w // 2
v = h // 2

#Max length of 2 finger joint
jid1 =  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
jid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")

r1 = model.jnt_range[jid1]
r2 = model.jnt_range[jid2]

finger1_max = r1[1] - r1[0]
finger2_max = r2[1] - r2[0]

#Max Length of Gripper
gripper_max_open  = finger1_max + finger2_max

#default gripper matrix
d_rotation = np.array([
    [1,0,0],
    [0,-1,0],
    [0,0,-1]
])

#Z-axis 90 rotation matrix
c, s = np.cos(np.pi/2), np.sin(np.pi/2)
z_90_rotation = np.array([
        [c,-s,0],
        [s,c,0],
        [0,0,1]
])

y_90_rotation = np.array([
    [c,0,s],
    [0,1,0],
    [-s,0,c]
])


long_rotated = d_rotation @ z_90_rotation
tall_rotated = d_rotation @ y_90_rotation

num_box = 0

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

params = {
    "wait":             {"alpha": 0.9,  "k_null": 0.3, "damping": 0.05},
    "start":            {"alpha": 0.9,  "k_null": 0, "damping": 0.05},
    "open_gripper":     {"alpha": 0.9,  "k_null": 0, "damping": 0.05},
    "descend_to_cube":  {"alpha": 0.3,  "k_null": 0.15, "damping": 0.01, "rot_weight": 5},
    "close_gripper":    {"alpha": 0.3,  "k_null": 0, "damping": 0.05},
    "lift":             {"alpha": 0.1, "k_null": 0.15, "damping": 0.01, "rot_weight": 5},
    "move":             {"alpha": 0.3,  "k_null": 0.15, "damping": 0.01, "rot_weight": 5},
    "drop":             {"alpha": 0.3,  "k_null": 0.15, "damping": 0.01, "rot_weight": 5},
    "rotate_check":     {"alpha": 0.1, "k_null": 0.3, "damping": 0.01, "rot_weight": 10},  # ← 추가
    "move_to_center":   {"alpha": 0.3,  "k_null": 0.15, "damping": 0.01, "rot_weight": 5},
    "release_gripper":  {"alpha": 0.7,  "k_null": 0, "damping": 0.05},
    "move_to_default":  {"alpha": 0.7,  "k_null": 0.15, "damping": 0.05},
    "move_to_start":    {"alpha": 0.7,  "k_null": 0.15, "damping": 0.05},
    "end":              {"alpha": 0.5,  "k_null": 0.15, "damping": 0.05},
    "move_to_above_cube":{"alpha": 0.3,  "k_null": 0, "damping": 0.05},
}

# space
space_id = model.body("target_space").id
start_pos_id = model.body("starting_space").id
start_pos = data.xpos[start_pos_id].copy()
target_space_pos = data.xpos[space_id].copy()

# gripper site
gripper_site_id = model.site("gripper").id

L_max = 0
R_max = 0
for i in range(model.ngeom):
    bodyid = model.geom(i).bodyid
    bodyname = model.body(bodyid).name
    if "left_finger" in bodyname:
        if model.geom(i).type == mujoco.mjtGeom.mjGEOM_BOX:
            size = model.geom(i).size
            pos = model.geom(i).pos
            y_end = abs(pos[1]) + size[1]
            L_max = max(L_max, y_end)

    if "right_finger" in bodyname:
        if model.geom(i).type == mujoco.mjtGeom.mjGEOM_BOX:
            size = model.geom(i).size
            pos = model.geom(i).pos
            y_end = abs(pos[1]) + size[1]
            R_max = max(R_max, y_end)


LEFT_FINGER_THICKNESS = L_max
RIGHT_FINGER_THICKNESS = R_max




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



long_rotated = d_rotation @ z_90_rotation

num_box = 0
placed_boxes_territory = []

valid_boxes = []     
target_box_id = None  
target_pack_pos = None
packing_result = []
areas = []
sorted_boxes = []
placed_boxes = []

initialized = False
t_rotation = d_rotation.copy()
captured_q_nominal = None
saved_q_nominal = None
collison_rotate = "safe"
grip_dir = "None"

fixed_box_xy = None
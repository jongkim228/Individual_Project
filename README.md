# Individual_Project
## Autonomous Packing with a 7-DOF Robot Arm

## Overview
>**Note:** This project is developed and tested on Linus. Running on Linux is recommended for best demostration.

This project implements an autonomous bin-packing pipeline for the Franka Emika Panda 7-DOF robot arm in MuJoCo simulation. Boxes detected in the starting space and it is placed in the target bin using knapsack packing sovler. Before placment, the system performs AABB gripper collision check to avoid colisions with previously placed boxes.

'''
Individual_Project/
├── src/
│   ├── main.py
│   ├── collsion.py
|   ├── detection.py
|   ├── init.py
|   ├── inverse_kinematics.py
|   ├── motions.py
|   └── packing.py
|
├── scene1.xml
├── scene2.xml
├── scene3.xml
└── README.md
'''

## Requirments
- Python
- MuJoCo
- packingsovler

## Setup
### 1. Clone the repository
```bash
git clone https://github.com/jongkim228/Individual_Project.git
```

### 2. Install Python dependencies
```bash
   pip install mujoco numpy opencv
```

### 3. Download MuJoCo mengagerie (Franka Emika Panda)
```bash
   git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

### 4. Copy scene files to test
```bash
   cp scene1.xml scene2.xml scene3.xml mujoco_menagerie/franka_emika_panda/
```
   > You can also create your own scene XML files in `mujoco_menagerie/franka_emika_panda/` and update the path in `init.py`
   > model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/<filename>.xml")

### 5. Build packingsolver
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel && cmake --install build --config Release --prefix install
```  

### 6. Run
```bash
python main.py
```
> **Note:** On macOS, use `mjpython` instead and update `detection.py`:
> In `objects_in_fov` function, change:
> ```python
> if z >= 0:
>     return False
> zz = -z
> ```
> to:
> ```python
> if z <= 0:
>     return False
> zz = z
> ```
> Then run:
> ```bash
> mjpython main.py
> ```


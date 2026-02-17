import numpy as np
import mujoco


def inverse_kinematics(model, data, gripper_site_id,t_position, t_rotation, arm_actuator_ids, exceeds_length, alpha = 0.1):

    mujoco.mj_kinematics(model,data)

    ee_position = data.site_xpos[gripper_site_id].copy()
    ee_rotation = data.site_xmat[gripper_site_id].reshape(3,3).copy()


    pos_err = t_position - ee_position

    pos_err[0] *= 1.0
    pos_err[1] *= 3.0
    pos_err[2] *= 1.0

    rot_err = t_rotation @ ee_rotation.T
    rot_vec = 0.5 * np.array([
    rot_err[2,1] - rot_err[1,2],
    rot_err[0,2] - rot_err[2,0],
    rot_err[1,0] - rot_err[0,1]
    ])
    

    err = np.concatenate([pos_err, rot_vec])
    
    #jacobian position
    jac_pos = np.zeros((3, model.nv))
    #jacobian rotation
    jac_rot = np.zeros((3, model.nv))
    #jacobian calculation
    mujoco.mj_jacSite(model,data,jac_pos, jac_rot, gripper_site_id)
    J = np.vstack([jac_pos, jac_rot])

    JJt = J @ J.T

    #pseudo inverse
    j_pse_inverse = J.T @ np.linalg.solve(JJt + 0.01 * np.eye(6), np.eye(6))

    dq_t = j_pse_inverse @ err

    q_nominal = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8])
    q_cur = data.qpos[:7].copy()

    N = np.eye(model.nv) - j_pse_inverse @ J

    k_null = 0.5
    dq0 = np.zeros(model.nv)
    dq0[:7] = (q_nominal - q_cur)

    dq = dq_t + k_null * (N @ dq0)

    q_target = q_cur + alpha * dq[:7]  
    data.ctrl[arm_actuator_ids] = q_target
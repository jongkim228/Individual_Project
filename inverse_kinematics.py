import numpy as np
import mujoco

def inverse_kinematics(model, data, gripper_site_id, t_position, t_rotation, 
                       arm_actuator_ids, alpha=0.3, k_null=0.05, damping=0.05):

    mujoco.mj_kinematics(model, data)

    q_nominal = np.array([0.0, -0.6, 0.0, -1.5, 0.0, 1.6, 0.8])
    actuator_weights = np.array([1, 0.5, 1, 0.5, 1, 0.5, 1])
    weights_inverse = np.diag(1 / actuator_weights)
    eomg = 0.001
    ev = 0.0001
    max_iter = 20

    ee_position = data.site_xpos[gripper_site_id].copy()
    ee_rotation = data.site_xmat[gripper_site_id].reshape(3, 3).copy()

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
    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, gripper_site_id)
    J = np.vstack([jac_pos, jac_rot])[:, :7]

    j_pse_inverse = weights_inverse @ J.T @ np.linalg.solve(
        J @ weights_inverse @ J.T + damping * np.eye(6), np.eye(6))

    q_cur = data.qpos[:7].copy()
    N = np.eye(7) - j_pse_inverse @ J
    dq0 = q_nominal - q_cur
    dq = (j_pse_inverse @ err) + k_null * (N @ dq0)

    q_target = q_cur + alpha * dq[:7]
    q_virtual = q_target.copy()


    for _ in range(max_iter):


        tmp_qpos = data.qpos[:7].copy()
        data.qpos[:7] = q_virtual
        mujoco.mj_forward(model, data)

        ee_position = data.site_xpos[gripper_site_id].copy()
        ee_rotation = data.site_xmat[gripper_site_id].reshape(3, 3).copy()

        pos_err = t_position - ee_position
        rot_err = t_rotation @ ee_rotation.T
        rot_vec = 0.5 * np.array([
            rot_err[2,1] - rot_err[1,2],
            rot_err[0,2] - rot_err[2,0],
            rot_err[1,0] - rot_err[0,1]
        ])
        err = np.concatenate([pos_err, rot_vec])

        data.qpos[:7] = tmp_qpos
        mujoco.mj_forward(model, data)

        if np.linalg.norm(rot_vec) < eomg and np.linalg.norm(pos_err) < ev:
            break

        data.qpos[:7] = q_virtual
        mujoco.mj_forward(model, data)

        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, gripper_site_id)
        J = np.vstack([jac_pos, jac_rot])[:, :7]

        j_pse_inverse = weights_inverse @ J.T @ np.linalg.solve(
            J @ weights_inverse @ J.T + damping * np.eye(6), np.eye(6))

        N = np.eye(7) - j_pse_inverse @ J
        dq0 = q_nominal - q_virtual
        dq = (j_pse_inverse @ err) + k_null * (N @ dq0)

        q_virtual += alpha * dq[:7]


        data.qpos[:7] = tmp_qpos
        mujoco.mj_forward(model, data)

    data.ctrl[arm_actuator_ids] = q_virtual
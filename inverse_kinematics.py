import numpy as np
import mujoco


def inverse_kinematics(model, data, gripper_site_id, t_position, t_rotation,
                       arm_actuator_ids, alpha=0.3, k_null=0.05, damping=0.05,
                       rot_weight=1.0, custom_q_nominal=None,
                       w_posture=1.0, w_limit=0.0):

    mujoco.mj_kinematics(model, data)

    if custom_q_nominal is not None:
        q_nominal = custom_q_nominal
    else:
        q_nominal = np.array([0.0, -0.6, 0.0, -1.5, 0.0, 1.6, 0.8])

    eomg = 0.001
    ev = 0.0001
    max_iter = 50

    q_min = model.jnt_range[:7, 0].copy()
    q_max = model.jnt_range[:7, 1].copy()

    def build_dq0(q):
        dq0 = w_posture * (q_nominal - q)
        if w_limit > 0.0:
            q_mid   = 0.5 * (q_min + q_max)
            q_range = np.where((q_max - q_min) > 1e-6, q_max - q_min, 1.0)
            grad_h1 = 2.0 * (q - q_mid) / (q_range ** 2)
            dq0 += w_limit * (-grad_h1)
        return dq0

    q_cur = data.qpos[:7].copy()
    original_qpos = q_cur.copy() 

    ee_position = data.site_xpos[gripper_site_id].copy()
    ee_rotation = data.site_xmat[gripper_site_id].reshape(3, 3).copy()
    pos_err = t_position - ee_position
    rot_err = t_rotation @ ee_rotation.T
    rot_vec = 0.5 * np.array([
        rot_err[2,1] - rot_err[1,2],
        rot_err[0,2] - rot_err[2,0],
        rot_err[1,0] - rot_err[0,1]
    ])
    err = np.concatenate([pos_err, rot_vec * rot_weight])

    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, gripper_site_id)
    J = np.vstack([jac_pos, jac_rot])[:, :7]
    j_pse_inverse = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), np.eye(6))

    N   = np.eye(7) - j_pse_inverse @ J
    dq0 = build_dq0(q_cur)
    dq  = (j_pse_inverse @ err) + k_null * (N @ dq0)
    q_virtual = q_cur + alpha * dq[:7]

    for _ in range(max_iter):
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
        err = np.concatenate([pos_err, rot_vec * rot_weight])

        if np.linalg.norm(rot_vec) < eomg and np.linalg.norm(pos_err) < ev:
            break

        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, gripper_site_id)
        J = np.vstack([jac_pos, jac_rot])[:, :7]
        j_pse_inverse = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), np.eye(6))

        N   = np.eye(7) - j_pse_inverse @ J
        dq0 = build_dq0(q_virtual)
        dq  = (j_pse_inverse @ err) + k_null * (N @ dq0)
        q_virtual += alpha * dq[:7]

    data.qpos[:7] = original_qpos
    mujoco.mj_forward(model, data)

    data.ctrl[arm_actuator_ids] = q_virtual
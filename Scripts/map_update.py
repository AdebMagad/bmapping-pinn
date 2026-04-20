# %%

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from numpy.linalg import inv
from scipy.linalg import cholesky
from collections import deque
import tensorflow as tf
import scipy.special as sc

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "text.latex.preamble": r"\usepackage{amsfonts}"
})

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


def circB(x, y, z):
    r_sq = x**2 + y**2 + z**2
    rho_sq = x**2 + y**2
    rho_sq = np.maximum(rho_sq, 1e-12)

    alpha_sq = 1.0 + r_sq - 2.0 * np.sqrt(rho_sq)
    beta_sq = 1.0 + r_sq + 2.0 * np.sqrt(rho_sq)
    beta_sq = np.maximum(beta_sq, 1e-12)
    k_sq = 1.0 - alpha_sq / beta_sq
    k_sq = np.clip(k_sq, 0.0, 1.0 - 1e-12)

    e_k_sq = sc.ellipe(k_sq)
    k_k_sq = sc.ellipk(k_sq)

    Bx = 2.0 * x * z / (alpha_sq * rho_sq * np.sqrt(beta_sq)) * (
        (1.0 + r_sq) * e_k_sq - alpha_sq * k_k_sq
    )

    if np.abs(x) < 1e-12:
        By = 2.0 * y * z / (alpha_sq * rho_sq * np.sqrt(beta_sq)) * (
            (1.0 + r_sq) * e_k_sq - alpha_sq * k_k_sq
        )
    else:
        By = y * Bx / x

    Bz = 2.0 / (alpha_sq * np.sqrt(beta_sq)) * (
        (1.0 - r_sq) * e_k_sq + alpha_sq * k_k_sq
    )

    return np.array([Bx, By, Bz]) * 50.0


def B(x, y, z):
    return (
        0.6 * circB(x + 1.01, y + 1.0, z - 4.0)
        + 0.2 * circB(x - 1.01, y - 1.0, z - 4.0)
        - 0.8 * circB(x + 1.01, y - 1.0, z - 4.0)
        - 0.5 * circB(x - 1.01, y + 1.0, z - 4.0)
        - 0.98 * circB(x + 1.01, y + 1.0, z + 4.0)
        - 0.46 * circB(x - 1.01, y - 1.0, z + 4.0)
        + 0.35 * circB(x + 1.01, y - 1.0, z + 4.0)
        + 0.87 * circB(x - 1.01, y + 1.0, z + 4.0)
    )


def quat_normalize(q):
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return q
    return q / norm


def quat2eul(q):
    q = quat_normalize(q)
    w, x, y, z = q.flatten()

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def eul2quat(eul):
    roll, pitch, yaw = eul.flatten()

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quat2Rot(q):
    q = quat_normalize(np.asarray(q).reshape(-1))
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),       1 - 2 * (x**2 + z**2),   2 * (y * z - x * w)],
        [2 * (x * z - y * w),       2 * (y * z + x * w),     1 - 2 * (x**2 + y**2)]
    ])
    return R



def eight_shaped_trajectory(T, N, A, B_amp, omega=0.15, z0=0.0):
    t = np.linspace(0, (N - 1) * T, N)
    x = A * np.sin(omega * t)
    y = B_amp * np.sin(omega * t) * np.cos(omega * t)
    z = np.full_like(x, z0)

    vx = np.gradient(x, T)
    vy = np.gradient(y, T)
    yaw = np.unwrap(np.arctan2(vy, vx))

    p = np.vstack((x, y, z))
    q = np.zeros((4, N))
    for k in range(N):
        q[:, k] = eul2quat(np.array([[0.0, 0.0, yaw[k]]]).T)

    p_0 = p[:, 0].copy()
    q_0 = q[:, 0].copy()
    return p, p_0, q, q_0, N



def Generate_control_signals_and_odometry(
    p_gt, q_gt, p_0, q_0, T, sigma_u=0.02, sigma_v=0.02, sigma_r=0.01
):
    N = p_gt.shape[1]

    xdot = np.gradient(p_gt[0, :], T)
    ydot = np.gradient(p_gt[1, :], T)

    euls = np.array([quat2eul(q_gt[:, k]) for k in range(N)]).T
    yaw = np.unwrap(euls[2, :])
    yawdot = np.gradient(yaw, T)

    cpsi = np.cos(yaw)
    spsi = np.sin(yaw)
    u_true = cpsi * xdot + spsi * ydot
    v_true = -spsi * xdot + cpsi * ydot
    r_true = yawdot

    u_meas = u_true + np.random.normal(0.0, sigma_u, size=N)
    v_meas = v_true + np.random.normal(0.0, sigma_v, size=N)
    r_meas = r_true + np.random.normal(0.0, sigma_r, size=N)

    nu_meas = np.vstack([u_meas, v_meas, r_meas])

    p_DR = np.zeros((3, N))
    q_DR = np.zeros((4, N))
    p_DR[:, 0] = p_0.reshape(3)
    q_DR[:, 0] = q_0.reshape(4)

    psi = quat2eul(q_0)[2]

    for k in range(N - 1):
        u, v, r = u_meas[k], v_meas[k], r_meas[k]
        xdot_k = u * np.cos(psi) - v * np.sin(psi)
        ydot_k = u * np.sin(psi) + v * np.cos(psi)
        psidot_k = r

        p_DR[0, k + 1] = p_DR[0, k] + xdot_k * T + np.random.normal(0.0, 0.005)
        p_DR[1, k + 1] = p_DR[1, k] + ydot_k * T + np.random.normal(0.0, 0.005)
        p_DR[2, k + 1] = p_DR[2, k]

        psi = psi + psidot_k * T
        q_nom = eul2quat(np.array([[0.0, 0.0, psi]]).T).flatten()
        q_DR[:, k + 1] = quat_normalize(q_nom + np.random.normal(0.0, 0.005, size=4))

    return nu_meas, p_DR, q_DR


def surface_ship_process_model(state, control, R_p, R_q, T):
    x, y, z, yaw, u, v = state
    u_m, v_m, r_m = control

    x_p1 = x + np.random.normal(0.0, R_p[0, 0]) + (u_m * np.cos(yaw) - v_m * np.sin(yaw)) * T
    y_p1 = y + np.random.normal(0.0, R_p[1, 1]) + (u_m * np.sin(yaw) + v_m * np.cos(yaw)) * T
    z_p1 = 0.0
    yaw_p1 = yaw + r_m * T + np.random.normal(0.0, R_q[2, 2])
    u_p1 = u_m * T + np.random.normal(0.0, R_q[1, 1])
    v_p1 = v_m * T + np.random.normal(0.0, R_q[0, 0])

    return np.array([x_p1, y_p1, z_p1, yaw_p1, u_p1, v_p1], dtype=float)


def magnetic_measurement_model(state, pinn_model):
    px, py, pz, yaw, _, _ = state
    inputs = tf.constant([[float(px), float(py), float(pz)]], dtype=tf.float32)
    B_world = pinn_model(inputs, training=False).numpy().flatten()

    q_i = eul2quat(np.array([[0.0, 0.0, yaw]]).T)
    R_bw = quat2Rot(q_i)
    z_pred = R_bw @ B_world
    return z_pred


def magnetic_measurement_model_true(state):
    px, py, pz, yaw, _, _ = state
    B_world = B(px, py, pz)
    q_i = eul2quat(np.array([[0.0, 0.0, yaw]]).T)
    R_bw = quat2Rot(q_i)
    z_pred = R_bw @ B_world
    return z_pred

def load_fresh_pinn_model(model_path):
    return tf.keras.models.load_model(model_path)


def yaw_to_rot_tf(yaw_batch):
    c = tf.cos(yaw_batch)
    s = tf.sin(yaw_batch)
    zeros = tf.zeros_like(c)
    ones = tf.ones_like(c)

    row1 = tf.stack([c, -s, zeros], axis=1)
    row2 = tf.stack([s,  c, zeros], axis=1)
    row3 = tf.stack([zeros, zeros, ones], axis=1)
    return tf.stack([row1, row2, row3], axis=1)


def sample_collocation_points(domain_bounds, n_colloc):
    x_min, x_max = domain_bounds["x"]
    y_min, y_max = domain_bounds["y"]
    z_min, z_max = domain_bounds["z"]

    x = np.random.uniform(x_min, x_max, size=(n_colloc, 1))
    y = np.random.uniform(y_min, y_max, size=(n_colloc, 1))
    z = np.random.uniform(z_min, z_max, size=(n_colloc, 1))
    pts = np.hstack([x, y, z]).astype(np.float32)
    return tf.convert_to_tensor(pts, dtype=tf.float32)


def body_field_prediction_tf(pinn_model, pos_batch, yaw_batch):
    B_world = pinn_model(pos_batch, training=True)
    R_batch = yaw_to_rot_tf(yaw_batch)
    B_body = tf.einsum("bij,bj->bi", R_batch, B_world)
    return B_body


def physics_losses_tf(pinn_model, colloc_points):
    with tf.GradientTape() as tape_phys:
        tape_phys.watch(colloc_points)
        B_pred = pinn_model(colloc_points, training=True)

    jac = tape_phys.batch_jacobian(B_pred, colloc_points)

    dBx_dx = jac[:, 0, 0]
    dBy_dy = jac[:, 1, 1]
    dBz_dz = jac[:, 2, 2]

    dBz_dy = jac[:, 2, 1]
    dBy_dz = jac[:, 1, 2]
    dBx_dz = jac[:, 0, 2]
    dBz_dx = jac[:, 2, 0]
    dBy_dx = jac[:, 1, 0]
    dBx_dy = jac[:, 0, 1]

    div_B = dBx_dx + dBy_dy + dBz_dz
    curl_x = dBz_dy - dBy_dz
    curl_y = dBx_dz - dBz_dx
    curl_z = dBy_dx - dBx_dy

    div_loss = tf.reduce_mean(tf.square(div_B))
    curl_loss = tf.reduce_mean(tf.square(curl_x) + tf.square(curl_y) + tf.square(curl_z))

    return div_loss, curl_loss


def online_map_update(
    pinn_model,
    optimizer,
    replay_buffer,
    corrected_state,
    z_meas,
    domain_bounds,
    lambda_phys=0.05,
    n_colloc=1,
    replay_batch_size=1,
    online_steps=1
):
    px, py, pz, yaw, _, _ = corrected_state

    replay_buffer.append({
        "pos": np.array([px, py, pz], dtype=np.float32),
        "yaw": np.float32(yaw),
        "z_meas": np.array(z_meas, dtype=np.float32),
    })

    if len(replay_buffer) == 0:
        return pinn_model


    for _ in range(online_steps):
        batch_size = min(replay_batch_size, len(replay_buffer))
        mini_batch = random.sample(replay_buffer, batch_size)

        pos_batch = tf.convert_to_tensor(
            np.stack([sample["pos"] for sample in mini_batch], axis=0),
            dtype=tf.float32
        )
        yaw_batch = tf.convert_to_tensor(
            np.array([sample["yaw"] for sample in mini_batch], dtype=np.float32),
            dtype=tf.float32
        )
        z_meas_batch = tf.convert_to_tensor(
            np.stack([sample["z_meas"] for sample in mini_batch], axis=0),
            dtype=tf.float32
        )

        colloc_points = sample_collocation_points(domain_bounds, n_colloc)


        #updating the pinn weights (map update of the SLAM system)
        with tf.GradientTape() as tape:
            z_pred_batch = body_field_prediction_tf(pinn_model, pos_batch, yaw_batch)
            data_loss = tf.reduce_mean(tf.square(z_pred_batch - z_meas_batch))
            div_loss, curl_loss = physics_losses_tf(pinn_model, colloc_points)
            total_loss = data_loss + lambda_phys * (div_loss + curl_loss)

        grads = tape.gradient(total_loss, pinn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, pinn_model.trainable_variables))

    return pinn_model


def UKF_surface_ship_localization(
    N,
    nu_meas,
    y_mag,
    q_0,
    p_0,
    P_0,
    R_p,
    R_q,
    sigma_y,
    pinn_model,
    T,
    domain_bounds,
    online_lr=1e-4,
    lambda_phys_online=0.05,
    n_colloc_online=1,
    replay_batch_size=1,
    replay_buffer_size=1,
    online_steps=1
):
    n = 6
    alpha = 0.5
    beta = 2.0
    kappa = 3 - n
    lambda_val = alpha**2 * (n + kappa) - n

    WM = np.zeros((2 * n + 1, 1))
    WC = np.zeros((2 * n + 1, 1))
    for j in range(2 * n + 1):
        if j == 0:
            WM[j, 0] = lambda_val / (n + lambda_val)
            WC[j, 0] = lambda_val / (n + lambda_val) + (1 - alpha**2 + beta)
        else:
            WM[j, 0] = 1.0 / (2.0 * (n + lambda_val))
            WC[j, 0] = WM[j, 0]

    q_UKF = np.zeros((4, N))
    p_UKF = np.zeros((3, N))

    yaw0 = quat2eul(q_0)[2]
    u0 = float(nu_meas[0, 0])
    v0 = float(nu_meas[1, 0])

    m_state = np.array([p_0[0], p_0[1], p_0[2], yaw0, u0, v0], dtype=float)

    P_UKF = P_0.copy()
    R_yawuv = np.diag([R_q[2, 2], R_q[0, 0], R_q[1, 1]])
    Q_UKF = np.block([
        [R_p, np.zeros((3, 3))],
        [np.zeros((3, 3)), R_yawuv]
    ])
    R_UKF = sigma_y**2 * np.eye(3)

    p_UKF[:, 0] = p_0.reshape(3)
    q_UKF[:, 0] = q_0.reshape(4)

    optimizer = tf.keras.optimizers.Adam(learning_rate=online_lr)
    replay_buffer = deque(maxlen=replay_buffer_size)

    for k in range(1, N):
        u_m = float(nu_meas[0, k - 1])
        v_m = float(nu_meas[1, k - 1])
        r_m = float(nu_meas[2, k - 1])

        P_UKF = 0.5 * (P_UKF + P_UKF.T)
        try:
            A = cholesky(P_UKF, lower=True)
        except Exception:
            A = cholesky(P_UKF + 1e-9 * np.eye(n), lower=True)

        SX = np.hstack((np.zeros((n, 1)), A, -A))
        SX = np.sqrt(n + lambda_val) * SX + m_state.reshape(-1, 1)

        HX = np.zeros_like(SX)
        for i in range(2 * n + 1):
            HX[:, i] = surface_ship_process_model(
                state=SX[:, i],
                control=[u_m, v_m, r_m],
                R_p=R_p,
                R_q=R_q,
                T=T
            )

        m_state = np.zeros((n,))
        for i in range(2 * n + 1):
            m_state += WM[i, 0] * HX[:, i]

        P_UKF = np.zeros((n, n))
        for i in range(2 * n + 1):
            dx = (HX[:, i] - m_state).reshape(-1, 1)
            P_UKF += WC[i, 0] * (dx @ dx.T)
        P_UKF += Q_UKF

        P_UKF = 0.5 * (P_UKF + P_UKF.T)
        try:
            A = cholesky(P_UKF, lower=True)
        except Exception:
            A = cholesky(P_UKF + 1e-9 * np.eye(n), lower=True)

        SX = np.hstack((np.zeros((n, 1)), A, -A))
        SX = np.sqrt(n + lambda_val) * SX + m_state.reshape(-1, 1)

        HY = np.zeros((3, 2 * n + 1))
        for i in range(2 * n + 1):
            HY[:, i] = magnetic_measurement_model(SX[:, i], pinn_model)

        mu = np.zeros((3, 1))
        for i in range(2 * n + 1):
            mu += WM[i, 0] * HY[:, i].reshape(-1, 1)

        S = R_UKF.copy()
        C = np.zeros((n, 3))
        for i in range(2 * n + 1):
            dy = HY[:, i].reshape(-1, 1) - mu
            dx = SX[:, i].reshape(-1, 1) - m_state.reshape(-1, 1)
            S += WC[i, 0] * (dy @ dy.T)
            C += WC[i, 0] * (dx @ dy.T)

        K = C @ inv(S)
        innovation = y_mag[:, k].reshape(-1, 1) - mu
        m_state = (m_state.reshape(-1, 1) + K @ innovation).flatten()
        P_UKF = P_UKF - K @ S @ K.T
        P_UKF = 0.5 * (P_UKF + P_UKF.T)

        p_UKF[:, k] = m_state[0:3]
        q_UKF[:, k] = eul2quat(np.array([[0.0, 0.0, m_state[3]]]).T).flatten()

        print(np.mean((HX[:, i] - m_state)))
        # if np.mean((HX[:, i] - m_state))>0.5:
        pinn_model = online_map_update(
            pinn_model=pinn_model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            corrected_state=m_state,
            z_meas=y_mag[:, k],
            domain_bounds=domain_bounds,
            lambda_phys=lambda_phys_online,
            n_colloc=n_colloc_online,
            replay_batch_size=replay_batch_size,
            online_steps=online_steps
        )

    return p_UKF, q_UKF, pinn_model



if __name__ == '__main__':
    print('Start simulations')
    tic = time.time()

    MODEL_PATH = "saved_models/trained_pinn_model"
    # MODEL_PATH = "saved_models/trained_plain_nn_model"

    # Load once just to validate path / availability
    _ = tf.keras.models.load_model(MODEL_PATH)

    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    T = 0.1
    N = 800
    experiments = 50

    p_gt, p_0_gt, q_gt, q_0_gt, N = eight_shaped_trajectory(
        T, N, A=1.0, B_amp=1.2, omega=0.15, z0=0.0
    )

    margin_xy = 0.25
    domain_bounds = {
        "x": (float(np.min(p_gt[0, :]) - margin_xy), float(np.max(p_gt[0, :]) + margin_xy)),
        "y": (float(np.min(p_gt[1, :]) - margin_xy), float(np.max(p_gt[1, :]) + margin_xy)),
        "z": (-0.25, 0.25),
    }

    error_UKFs, error_DRs = [], []
    q_error_UKFs, q_error_DRs = [], []

    initial_error = 0.1
    P_0_cov = 0.0001 * np.eye(6)
    P_0_cov[0:2, 0:2] = initial_error * np.eye(2)

    R_p = 0.0001 * np.eye(3) * T
    R_q = 0.0001 * np.eye(3) * T

    sigma_u, sigma_v, sigma_r = 0.01, 0.01, 0.01
    sigma_y_meas = 0.01
    sigma_y_filter = 0.98

    p_0 = p_0_gt + np.array([
        np.sqrt(0.5) * initial_error,
        np.sqrt(0.5) * initial_error,
        0.0
    ])
    q_0 = q_0_gt.copy()

    tt = np.zeros(experiments)

    nu_meas, p_DR, q_DR = Generate_control_signals_and_odometry(
        p_gt=p_gt, q_gt=q_gt, p_0=p_0, q_0=q_0, T=T,
        sigma_u=sigma_u, sigma_v=sigma_v, sigma_r=sigma_r
    )

    updated_pinn_model = None

    for exp_idx in range(1, experiments + 1):
        t0 = time.perf_counter()

        # Fresh subclassed model instance for each experiment
        pinn_model = load_fresh_pinn_model(MODEL_PATH)

        p_dy = np.zeros((3, N))
        q_dy = np.zeros((4, N))
        p_dy[:, 0] = p_0.reshape(3)
        q_dy[:, 0] = q_0.reshape(4)

        y_mag = np.zeros((3, N))
        psi = quat2eul(q_dy[:, 0])[2]

        init_state = np.array([p_dy[0, 0], p_dy[1, 0], p_dy[2, 0], psi, 0.0, 0.0], dtype=float)
        y_mag[:, 0] = magnetic_measurement_model_true(init_state) + np.random.multivariate_normal(
            np.zeros(3), (sigma_y_meas ** 2) * np.eye(3)
        )

        for t in range(N - 1):
            state = np.array([p_dy[0, t], p_dy[1, t], p_dy[2, t], psi, 0.0, 0.0], dtype=float)
            u, v, r = nu_meas[0, t], nu_meas[1, t], nu_meas[2, t]
            state_t = surface_ship_process_model(state, [u, v, r], R_p, R_q, T)

            p_dy[:, t + 1] = state_t[0:3]
            psi = state_t[3]
            q_dy[:, t + 1] = eul2quat(np.array([[0.0, 0.0, psi]]).T)

            y_pred_true = magnetic_measurement_model_true(state_t)
            noise = np.random.multivariate_normal(np.zeros(3), (sigma_y_meas ** 2) * np.eye(3))
            y_mag[:, t + 1] = y_pred_true + noise

        p_UKF, q_UKF, updated_pinn_model = UKF_surface_ship_localization(
            N=N,
            nu_meas=nu_meas,
            y_mag=y_mag,
            q_0=q_0,
            p_0=p_0,
            P_0=P_0_cov,
            R_p=R_p,
            R_q=R_q,
            sigma_y=sigma_y_filter,
            pinn_model=pinn_model,
            T=T,
            domain_bounds=domain_bounds,
            online_lr=1e-4,
            lambda_phys_online=0.05,
            n_colloc_online=1,
            replay_batch_size=1,
            replay_buffer_size=1,
            online_steps=1
        )

        error_UKFs.append(p_dy - p_UKF)
        error_DRs.append(p_dy - p_DR)
        q_error_UKFs.append(q_dy - q_UKF)
        q_error_DRs.append(q_dy - q_DR)

        t1 = time.perf_counter()
        tt[exp_idx - 1] = t1 - t0
        print(f'Experiment {exp_idx}/{experiments} complete')
        print(f"Time spent: {tt[exp_idx - 1]:.4f} s")

    error_UKFs = np.array(error_UKFs)
    error_DRs = np.array(error_DRs)
    q_error_UKFs = np.array(q_error_UKFs)
    q_error_DRs = np.array(q_error_DRs)

    rmse_UKF = np.sqrt(np.sum(np.square(error_UKFs), axis=(0, 1)) / experiments)
    rmse_DR = np.sqrt(np.sum(np.square(error_DRs), axis=(0, 1)) / experiments)
    rmse_UKF_q = np.sqrt(np.sum(np.square(q_error_UKFs), axis=(0, 1)) / experiments)
    rmse_DR_q = np.sqrt(np.sum(np.square(q_error_DRs), axis=(0, 1)) / experiments)

    q_ukf_eul = np.zeros((3, q_UKF.shape[1]))
    q_dr_eul = np.zeros((3, q_UKF.shape[1]))
    q_dy_eul = np.zeros((3, q_UKF.shape[1]))

    for i in range(q_UKF.shape[1]):
        q_ukf_eul[:, i] = quat2eul(q_UKF[:, i])
        q_dr_eul[:, i] = quat2eul(q_DR[:, i])
        q_dy_eul[:, i] = quat2eul(q_dy[:, i])




    # ----------------------------------- Plots -----------------------------------
    plt.figure()
    plt.plot(range(N), rmse_DR, linewidth=1.5)
    plt.plot(range(N), rmse_UKF, linewidth=1.5)
    plt.xlabel(r'$t$ (steps)')
    plt.ylabel('RMSE (m)')
    plt.title(f'Position RMSE Across {experiments} Experiments')
    plt.legend(['Position Odometry RMSE', 'Position UKF RMSE'])
    plt.grid(True)
    plt.ylim(0, 0.5)
    plt.tight_layout()
    plt.savefig('fig5.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(range(N), rmse_DR_q, linewidth=1.5)
    plt.plot(range(N), rmse_UKF_q, linewidth=1.5)
    plt.xlabel(r'$t$ (steps)')
    plt.ylabel('RMSE (quaternion units)')
    plt.title(f'Orientation RMSE Across {experiments} Experiments')
    plt.legend(['Orientation Odometry RMSE', 'Orientation UKF RMSE'])
    plt.grid(True)
    plt.ylim(0, 0.15)
    plt.tight_layout()
    plt.savefig('fig4.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(p_UKF[0, :], linewidth=1.4)
    plt.plot(p_dy[0, :], linewidth=1.4)
    plt.title(r'$x$ and $y$ States Motion Visualization')
    plt.xlabel(r'$t$ (steps)')
    plt.ylabel(r'$x$ (m)')
    plt.legend([r'$x$-UKF', r'$x$-Odometry'])
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(p_UKF[1, :], linewidth=1.4)
    plt.plot(p_dy[1, :], linewidth=1.4)
    plt.xlabel(r'$t$ (steps)')
    plt.ylabel(r'$y$ (m)')
    plt.legend([r'$y$-UKF', r'$y$-Odometry'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fig3.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(q_ukf_eul[2, :], linewidth=1.4)
    plt.plot(q_dr_eul[2, :], linewidth=1.4)
    plt.title('Yaw angle over time')
    plt.xlabel(r'$t$ (steps)')
    plt.ylabel('Orientation (rad)')
    plt.legend(['Orientation-UKF', 'Orientation-Odometry'])
    plt.grid(True)
    plt.savefig('fig2.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(p_UKF[0, :], p_UKF[1, :], linewidth=1.2)
    plt.plot(p_dy[0, :], p_dy[1, :], linewidth=1.2)
    plt.title('Vehicle Motion Visualization')
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$y$ (m)')
    plt.legend(['UKF', 'Odometry'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fig1.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    sc_plot = plt.scatter(
        p_dy[0, :], p_dy[1, :],
        c=np.linalg.norm(y_mag, axis=0),
        cmap='turbo', s=20
    )
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$y$ (m)')
    plt.title('Variation of the estimated magnetic field values')
    cbar = plt.colorbar(sc_plot)
    cbar.set_label('Magnetic Field Values')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('mag_field.pdf', bbox_inches='tight')
    plt.show()

    print("Simulation completed.")
    timespent = time.time() - tic
    savemat('timespent.mat', {'timespent': timespent})

    workspace = {
        'p_gt': p_gt,
        'q_gt': q_gt,
        'p_UKF': p_UKF,
        'p_DR': p_DR,
        'q_UKF': q_UKF,
        'q_DR': q_DR,
        'y_mag': y_mag,
        'sigma_y_meas': sigma_y_meas,
        'sigma_y_filter': sigma_y_filter,
        'R_p': R_p,
        'R_q': R_q,
        'P_0_cov': P_0_cov,
        'p_0': p_0,
        'q_0': q_0,
        'N': N,
        'experiments': experiments,
        'nu_meas': nu_meas,
        'tt': tt,
        'rmse_UKF': rmse_UKF,
        'rmse_DR': rmse_DR,
        'rmse_UKF_q': rmse_UKF_q,
        'rmse_DR_q': rmse_DR_q,
        'domain_bounds_x': np.array(domain_bounds["x"]),
        'domain_bounds_y': np.array(domain_bounds["y"]),
        'domain_bounds_z': np.array(domain_bounds["z"]),
    }
    savemat('Workspace.mat', workspace)

    if updated_pinn_model is not None:
        updated_pinn_model.save("saved_models/trained_pinn_model_online_updated")

# %%
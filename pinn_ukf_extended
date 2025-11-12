# %%

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat, loadmat
from numpy.linalg import inv
from scipy.linalg import cholesky
import tensorflow as tf
import scipy.special as sc
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

def circB(x,y,z):
    #Defining some parameters to be used in the formulas
    r_sq = x**2 + y**2 + z**2
    rho_sq = x**2 + y**2
    alpha_sq = 1. + r_sq - 2. * np.sqrt(rho_sq)
    beta_sq = 1. + r_sq + 2. * np.sqrt(rho_sq)
    k_sq = 1. - alpha_sq/beta_sq

    #Evaluate elliptic integrals
    e_k_sq = sc.ellipe(k_sq)
    k_k_sq = sc.ellipk(k_sq)
    
    #Magnetic field components in Cartesian coordinates
    Bx = 2. * x * z / (alpha_sq * rho_sq * np.sqrt(beta_sq)) * ((1. + r_sq) * e_k_sq - alpha_sq * k_k_sq)
    By = y * Bx / x
    Bz = 2. / (alpha_sq * np.sqrt(beta_sq)) * ((1. - r_sq) * e_k_sq + alpha_sq * k_k_sq)
    
    return np.array([Bx,By,Bz]) * 50

# Now calculate the mag field of collection of circles
def B(x,y,z):
    return 0.6*circB(x + 1.01,y + 1.0,z - 4.0) + 0.2*circB(x - 1.01,y - 1.0, z - 4.0) - 0.8*circB(x + 1.01,y - 1.0,z - 4.0) - 0.5*circB(x - 1.01,y + 1.0,z - 4.0) - 0.98*circB(x + 1.01,y + 1.0,z + 4.0) - 0.46*circB(x - 1.01,y - 1.0,z + 4.0) + 0.35*circB(x + 1.01,y - 1.0,z + 4.0) + 0.87*circB(x - 1.01,y + 1.0,z + 4.0)

def quat_normalize(q):
    norm = np.linalg.norm(q)
    if norm == 0:
        return q
    return q / norm

def quat2eul(q):
    # q assumed in [w, x, y, z] format
    q = quat_normalize(q)
    w, x, y, z = q.flatten()
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi/2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # yaw (z-axis rotation)
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
    # Convert quaternion (with scalar first) to a rotation matrix.
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(y**2+z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2+z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2+y**2)]
    ])
    return R

def quatprod(q, r):
    # Quaternion product of q and r.
    w = q[0] * r[0] - np.dot(q[1:], r[1:])
    xyz = q[0] * r[1:] + r[0] * q[1:] + np.cross(q[1:], r[1:])
    return np.concatenate(([w], xyz))

def quat2angleaxis(q):
    eps = 1e-12
    norm_q = np.linalg.norm(q)
    if norm_q == 0:
        return np.zeros(3)
    q = q / norm_q
    angle = 2 * np.arccos(q[0])
    sin_term = np.sqrt(1 - q[0]**2)
    if sin_term < eps:
        return np.zeros(3)
    axis = q[1:] / sin_term
    return axis * angle

def exp_q_L(delta, q):
    # Left multiply by exp(delta) on SO(3)
    theta = np.linalg.norm(delta)
    eps = 1e-12
    if theta < eps:
        dq = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        dq = np.concatenate(([np.cos(theta / 2.0)], np.sin(theta / 2.0) * (delta / theta)))
    return quatprod(dq, q)

def quatprod(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat2angleaxis(q):
    if q[0] > 1.0:
        q = q / np.linalg.norm(q)
    angle = 2 * np.arccos(q[0])
    s = np.sqrt(1 - q[0]**2)
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = q[1:4] / s
    return angle * axis

# ---------------- 8-shaped Ground Truth (unchanged) ----------------

def eight_shaped_trajectory(T, N, A, B, omega=0.15, z0=0.0):

    t = np.linspace(0, (N - 1) * T, N)
    x = A * np.sin(omega * t)
    y = B * np.sin(omega * t) * np.cos(omega * t)
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


def Generate_control_signals_and_odometry(p_gt, q_gt, T, sigma_u=0.02, sigma_v=0.02, sigma_r=0.01):

    N = p_gt.shape[1]

    # World-frame velocities
    xdot = np.gradient(p_gt[0, :], T)
    ydot = np.gradient(p_gt[1, :], T)

    # Yaw from GT quaternions
    euls = np.array([quat2eul(q_gt[:, k]) for k in range(N)]).T  # 3xN
    yaw = np.unwrap(euls[2, :])
    yawdot = np.gradient(yaw, T)

    # World->Body mapping to get (u, v)
    cpsi = np.cos(yaw)
    spsi = np.sin(yaw)
    u_true =  cpsi * xdot + spsi * ydot
    v_true = -spsi * xdot + cpsi * ydot
    r_true = yawdot

    # Add noise (measured "odometry-like")
    u_meas = u_true + np.random.normal(0.0, sigma_u, size=N)
    v_meas = v_true + np.random.normal(0.0, sigma_v, size=N)
    r_meas = r_true + np.random.normal(0.0, sigma_r, size=N)

    nu_meas = np.vstack([u_meas, v_meas, r_meas])

    # Dead-reckoning integration 
    
    p_DR = np.zeros((3, N)) 
    q_DR = np.zeros((4, N)) 
    p_DR[:, 0] = p_0.reshape(3) 
    q_DR[:, 0] = q_0.reshape(4) 
    psi = yaw[0] 
    for k in range(N - 1): 
        u, v, r = u_meas[k], v_meas[k], r_meas[k] 
        xdot_k = u * np.cos(psi) - v * np.sin(psi) 
        ydot_k = u * np.sin(psi) + v * np.cos(psi) 
        psidot_k = r 
        p_DR[0, k+1] = p_DR[0, k] + xdot_k * T + np.random.normal(0.0, 0.005)
        p_DR[1, k+1] = p_DR[1, k] + ydot_k * T + np.random.normal(0.0, 0.005)
        p_DR[2, k+1] = p_DR[2, k] # z unchanged 
        psi = psi + psidot_k * T 
        q_DR[:, k+1] = eul2quat(np.array([[0.0, 0.0, psi]]).T) + + np.random.normal(0.0, 0.005, size=4)
    
    return nu_meas, p_DR, q_DR


def surface_ship_process_model(state, control,R_p, R_q, T):

    x, y, z, yaw, u, v = state
    u_m, v_m, r_m = control

    x_p1   = x + np.random.normal(0.0, R_p[0,0])+ (u_m * np.cos(yaw) - v_m * np.sin(yaw)) * T
    y_p1   = y + np.random.normal(0.0, R_p[1,1])+ (u_m * np.sin(yaw) + v_m * np.cos(yaw)) * T
    z_p1   = 0.0
    yaw_p1 = yaw + r_m * T + np.random.normal(0.0, R_q[2,2])  # simple euler integration
    u_p1   = u_m* T + np.random.normal(0.0, R_q[1,1])  # assume perfect tracking of commanded surge
    v_p1   = v_m* T + np.random.normal(0.0, R_q[0,0])  # assume perfect tracking of commanded sway

    return np.array([x_p1, y_p1, z_p1, yaw_p1, u_p1, v_p1], dtype=float)


def magnetic_measurement_model(state, pinn_model):
    px, py, pz, yaw, _, _ = state
    inputs = tf.constant([[float(px), float(py), float(pz)]], dtype=tf.float32)
    B_world = pinn_model.call(inputs).numpy().flatten()  # (3,)

    # Orientation: roll=pitch=0, yaw from state
    q_i = eul2quat(np.array([[0.0, 0.0, yaw]]).T)
    R_bw = quat2Rot(q_i)  # rotation into body frame
    z_pred = R_bw @ B_world  # body-frame magnetic field
    return z_pred  # (3,)


def UKF_surface_ship_localization(
    N, nu_meas, y_mag, q_0, p_0, P_0, R_p, R_q, sigma_y, pinn_model, T):

    n = 6  # [x,y,z,yaw,u,v]
    alpha = 0.5
    beta  = 2.0
    kappa = 3 - n
    lambda_val = alpha**2 * (n + kappa) - n

    WM = np.zeros((2*n+1, 1))
    WC = np.zeros((2*n+1, 1))
    for j in range(2*n+1):
        if j == 0:
            WM[j, 0] = lambda_val / (n + lambda_val)
            WC[j, 0] = lambda_val / (n + lambda_val) + (1 - alpha**2 + beta)
        else:
            WM[j, 0] = 1.0 / (2.0 * (n + lambda_val))
            WC[j, 0] = WM[j, 0]


    q_UKF = np.zeros((4, N))
    p_UKF = np.zeros((3, N))

    # Initial yaw from q_0
    yaw0 = quat2eul(q_0)[2]
    # Initialize u, v with first measurements
    u0 = float(nu_meas[0, 0])
    v0 = float(nu_meas[1, 0])

    # State: [x, y, z, yaw, u, v]
    m_state = np.array([p_0[0], p_0[1], p_0[2], yaw0, u0, v0], dtype=float)

    P_UKF = P_0.copy()  # should be 6x6
    # Map R_q (3x3) into yaw,u,v process noise (use diagonals)
    R_yawuv = np.diag([R_q[2,2], R_q[0,0], R_q[1,1]])
    Q_UKF = np.block([
        [R_p,                       np.zeros((3,3))],
        [np.zeros((3,3)),           R_yawuv        ]
    ])
    R_UKF = sigma_y**2 * np.eye(3)

    p_UKF[:, 0] = p_0.reshape(3)
    q_UKF[:, 0] = q_0.reshape(4)

    for k in range(1, N):
        # Controls from body-velocity measurements at step k-1
        u_m = float(nu_meas[0, k-1])
        v_m = float(nu_meas[1, k-1])
        r_m = float(nu_meas[2, k-1])

        # ---- Sigma point generation (prior)
        P_UKF = 0.5*(P_UKF + P_UKF.T)
        try:
            A = cholesky(P_UKF, lower=True)
        except Exception:
            A = cholesky(P_UKF + 1e-9*np.eye(n), lower=True)

        SX = np.hstack((np.zeros((n, 1)), A, -A))
        SX = np.sqrt(n + lambda_val) * SX + m_state.reshape(-1, 1)

        # ---- SURFACE SHIP PROCESS MODEL (refactored) ----
        HX = np.zeros_like(SX)
        for i in range(2*n+1):
            HX[:, i] = surface_ship_process_model(
                state=SX[:, i],
                control=[u_m, v_m, r_m],R_p=R_p, R_q=R_q,
                T=T
            )

        # ---- Predicted mean & covariance
        m_state = np.zeros((n,))
        for i in range(2*n+1):
            m_state += WM[i, 0] * HX[:, i]
        P_UKF = np.zeros((n, n))
        for i in range(2*n+1):
            dx = (HX[:, i] - m_state).reshape(-1,1)
            P_UKF += WC[i, 0] * (dx @ dx.T)
        P_UKF += Q_UKF

        # ---- Sigma points for measurement step
        P_UKF = 0.5*(P_UKF + P_UKF.T)
        try:
            A = cholesky(P_UKF, lower=True)
        except Exception:
            A = cholesky(P_UKF + 1e-9*np.eye(n), lower=True)
        SX = np.hstack((np.zeros((n, 1)), A, -A))
        SX = np.sqrt(n + lambda_val) * SX + m_state.reshape(-1, 1)

        # ---- Predict magnetic measurements for each sigma point (refactored)
        HY = np.zeros((3, 2*n+1))
        for i in range(2*n+1):
            HY[:, i] = magnetic_measurement_model(SX[:, i], pinn_model)

        # ---- Measurement mean & covariances
        mu = np.zeros((3, 1))
        for i in range(2*n+1):
            mu += WM[i, 0] * HY[:, i].reshape(-1,1)

        S = R_UKF.copy()
        C = np.zeros((n, 3))
        for i in range(2*n+1):
            dy = HY[:, i].reshape(-1,1) - mu
            dx = SX[:, i].reshape(-1,1) - m_state.reshape(-1,1)
            S += WC[i, 0] * (dy @ dy.T)
            C += WC[i, 0] * (dx @ dy.T)

        # ---- Update
        K = C @ inv(S)
        innovation = y_mag[:, k].reshape(-1,1) - mu
        m_state = (m_state.reshape(-1,1) + K @ innovation).flatten()
        P_UKF = P_UKF - K @ S @ K.T
        P_UKF = 0.5*(P_UKF + P_UKF.T)

        # Save filtered trajectory
        p_UKF[:, k] = m_state[0:3]
        q_UKF[:, k] = eul2quat(np.array([[0.0, 0.0, m_state[3]]]).T).flatten()

    return p_UKF, q_UKF



if __name__ == '__main__':
    print('Start simulations')
    tic = time.time()

    # Load trained PINN model (update path as needed)
    MODEL_PATH = "saved_models/trained_pinn_model"
    # MODEL_PATH = "saved_models/trained_plain_nn_model"
    pinn_model = tf.keras.models.load_model(MODEL_PATH)

    SEED = 42
    np.random.seed(SEED)
    T = 0.1                # time step [s]
    N = 800                # number of steps
    experiments = 50       # Monte Carlo runs

    # Generate an 8-shaped ground-truth trajectory
    p, p_0, q, q_0, N = eight_shaped_trajectory(T, N, A=1, B=1.2, omega=0.15, z0=0.0)

    # Lists to accumulate errors over experiments
    error_UKFs, error_DRs = [], []
    q_error_UKFs, q_error_DRs = [], []

    # Initial covariance and noise params
    initial_error = 0.1
    P_0_cov = 0.0001 * np.eye(6)              
    P_0_cov[0:2, 0:2] = initial_error * np.eye(2)
    R_p = 0.0001 * np.eye(3) * T             # process noise for position
    R_q = 0.0001 * np.eye(3) * T             # process noise proxy for [yaw,u,v]

    # Odometry noise (body velocities)
    sigma_u, sigma_v, sigma_r = 0.01, 0.01, 0.01

    # Measurement noise (separate variables for clarity)
    sigma_y_meas = 0.01   # used to synthesize magnetometer measurements
    sigma_y_filter = 0.98  # assumed by the filter (often > meas noise)
    p_0 = p_0 + np.array([np.sqrt(0.5) * initial_error,
                            np.sqrt(0.5) * initial_error, 0.0])
    p_dy = np.zeros((3, N))
    q_dy = np.zeros((4, N))
    p_dy[:, 0] = p_0.reshape(3)
    q_dy[:, 0] = q_0.reshape(4)
    tt = np.zeros(experiments)
    nu_meas, p_DR, q_DR = Generate_control_signals_and_odometry(p_gt=p, q_gt=q, T=T, sigma_u=sigma_u, sigma_v=sigma_v,sigma_r=sigma_r)
    for exp_idx in range(1, experiments + 1):
        t0 = time.perf_counter()
        y_mag = np.zeros((3, N))
        psi = quat2eul(q_dy[:, 0])[2]
        for t in range(N-1):
            state = np.array([p_dy[0, t], p_dy[1, t], p_dy[2, t], psi, 0.0, 0.0], dtype=float)
            u, v, r = nu_meas[0, t], nu_meas[1, t], nu_meas[2, t]
            state_t = surface_ship_process_model(state, [u, v, r],R_p, R_q, T)
            p_dy[:, t+1] = state_t[0:3]
            # p_dy[2, t+1] = 0.0
            psi = state_t[3]
            q_dy[:, t+1] = eul2quat(np.array([[0.0, 0.0, psi]]).T)
            state_t = np.array([p_dy[0, t+1], p_dy[1, t+1], p_dy[2, t+1], psi, 0.0, 0.0], dtype=float)
            y_pred = magnetic_measurement_model(state_t, pinn_model)
            noise = np.random.multivariate_normal(np.zeros(3), (sigma_y_meas ** 2) * np.eye(3))
            y_mag[:, t] = y_pred + noise
        

        p_UKF, q_UKF = UKF_surface_ship_localization(
            N=N, nu_meas=nu_meas, y_mag=y_mag, q_0=q_0, p_0=p_0,
            P_0=P_0_cov, R_p=R_p, R_q=R_q, sigma_y=sigma_y_filter,
            pinn_model=pinn_model, T=T
        )
        
        
        error_UKFs.append(p_dy - p_UKF)
        error_DRs.append(p_dy - p_DR)
        q_error_UKFs.append(q_dy - q_UKF)
        q_error_DRs.append(q_dy - q_DR)
        t1 = time.perf_counter()
        tt[exp_idx - 1] = t1 - t0
        print(f'Experiment {exp_idx}/{experiments} complete')
        print(f"Time spent: {tt[exp_idx - 1]:.4f} s")
    error_UKFs = np.array(error_UKFs)   # (E, 3, N)
    error_DRs  = np.array(error_DRs)    # (E, 3, N)
    q_error_UKFs = np.array(q_error_UKFs)
    q_error_DRs  = np.array(q_error_DRs)

    rmse_UKF = np.sqrt(np.sum(np.square(error_UKFs), axis=(0, 1)) / experiments)
    rmse_DR  = np.sqrt(np.sum(np.square(error_DRs), axis=(0, 1)) / experiments)
    rmse_UKF_q = np.sqrt(np.sum(np.square(q_error_UKFs), axis=(0, 1)) / experiments)
    rmse_DR_q  = np.sqrt(np.sum(np.square(q_error_DRs),  axis=(0, 1)) / experiments)

    q_ukf_eul = np.zeros((3, q_UKF.shape[1]))
    q_dr_eul  = np.zeros((3, q_UKF.shape[1]))
    q_dy_eul  = np.zeros((3, q_UKF.shape[1]))

    for i in range(q_UKF.shape[1]):
        q_ukf_eul[:, i] = quat2eul(q_UKF[:, i])
        q_dr_eul[:, i]  = quat2eul(q_DR[:, i])
        q_dy_eul[:, i]  = quat2eul(q_dy[:, i])

    # ----------------------------------- Plots -----------------------------------

    plt.figure()
    plt.plot(range(N), rmse_DR,  linewidth=1.5)
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
    plt.plot(range(N), rmse_DR_q,  linewidth=1.5)
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
    plt.plot(p_dy[0, :],     linewidth=1.4)
    plt.title(r'$x$ and $y$ States Motion Visualization')
    plt.xlabel(r'$t$ (steps)')
    plt.ylabel(r'$x$ (m)')
    plt.legend([r'$x$-UKF', r'$x$-Odometry'])
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(p_UKF[1, :], linewidth=1.4)
    plt.plot(p_dy[1, :],     linewidth=1.4)
    plt.xlabel(r'$t$ (steps)')
    plt.ylabel(r'$y$ (m)')
    plt.legend([r'$y$-UKF', r'$y$-Odometry'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fig3.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(q_ukf_eul[2, :], linewidth=1.4)
    plt.plot(q_dr_eul[2, :],  linewidth=1.4)
    plt.title('Yaw angle over time')
    plt.xlabel(r'$t$ (steps)')
    plt.ylabel('Orientation (rad)')
    plt.legend(['Orientation-UKF', 'Orientation-Odometry'])
    plt.grid(True)
    plt.savefig('fig2.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(p_UKF[0, :], p_UKF[1, :], linewidth=1.2)
    plt.plot(p_dy[0, :],  p_dy[1, :],  linewidth=1.2)
    plt.title('Vehicle Motion Visualization')
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$y$ (m)')
    plt.legend(['UKF', 'Odometry'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fig1.pdf', bbox_inches='tight')
    plt.show()

    
    plt.figure()
    sc = plt.scatter(p_dy[0,:], p_dy[1,:], c=np.linalg.norm(y_mag, axis=0), cmap='turbo', s=20)
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$y$ (m)')
    plt.title('Variation of the estimated magnetic field values')
    cbar = plt.colorbar(sc)
    cbar.set_label('Magnetic Field Values')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('mag_field.pdf', bbox_inches='tight')
    plt.show()
    # -------------------------------- Save artifacts --------------------------------
    print("Simulation completed.")
    timespent = time.time() - tic
    savemat('timespent.mat', {'timespent': timespent})

    # Save workspace (from last experiment where variables are defined)
    workspace = {
        'p': p, 'q': q, 'p_UKF': p_UKF,
        'p_DR': p_DR, 'q_UKF': q_UKF, 'q_DR': q_DR, 'y_mag': y_mag,
        'sigma_y_meas': sigma_y_meas, 'sigma_y_filter': sigma_y_filter,
        'R_p': R_p, 'R_q': R_q, 'P_0_cov': P_0_cov, 'p_0': p_0,
        'q_0': q_0, 'N': N, 'experiments': experiments,
        'q_0_gt': q_0, 'nu_meas': nu_meas
    }
    savemat('Workspace.mat', workspace)




# %%

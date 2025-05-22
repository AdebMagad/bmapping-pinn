# %%

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat
from numpy.linalg import inv
from scipy.linalg import cholesky
import tensorflow as tf
import scipy.special as sc
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
    # use 90 degrees if out of range
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

def rectangular_walk_ground_truth(T, xl, xu, yl, yu, margin, z, v):
    length_val = xu - xl - 2 * margin
    
    #height - Height of trajectory in meter
    height_val = yu - yl - 2 * margin
    
    #Distance traversed per sample
    diff_p = v * T
    
    # Determine the number of steps for the horizontal and vertical edges
    n_steps_length = int(np.floor(length_val / diff_p))
    n_steps_height = int(np.floor(height_val / diff_p))
    
    #Make the four edges of the rectangle
    # p1: from left to right along the bottom edge
    p1_vals = np.arange(diff_p, (n_steps_length + 0.5) * diff_p, diff_p)
    p1 = np.vstack((p1_vals, np.zeros(n_steps_length), np.zeros(n_steps_length)))
    
    # p2: from bottom to top along the right edge, shifted by the last point of p1
    p2_vals = np.arange(diff_p, (n_steps_height + 0.5) * diff_p, diff_p)
    p2 = np.vstack((np.zeros(n_steps_height), p2_vals, np.zeros(n_steps_height)))
    p2 = p2 + p1[:, -1].reshape(3, 1)
    
    # p3: from right to left along the top edge, shifted by the last point of p2
    p3_vals = -np.arange(diff_p, (n_steps_length + 0.5) * diff_p, diff_p)
    p3 = np.vstack((p3_vals, np.zeros(n_steps_length), np.zeros(n_steps_length)))
    p3 = p3 + p2[:, -1].reshape(3, 1)
    
    # p4: from top to bottom along the left edge, shifted by the last point of p3
    p4_vals = -np.arange(diff_p, (n_steps_height + 0.5) * diff_p, diff_p)
    p4 = np.vstack((np.zeros(n_steps_height), p4_vals, np.zeros(n_steps_height)))
    p4 = p4 + p3[:, -1].reshape(3, 1)
    
    #Concatenate the edges to make full rectangle
    p = np.concatenate((p1, p2, p3, p4), axis=1)
    
    #Repeat the rectangular walking pattern to make four laps
    p = np.concatenate((p, p, p, p), axis=1)
    
    #Shift all positions according to the initial position
    shift_vector = np.array([[xl + margin], [yl + margin], [z]])
    p = p + shift_vector
    p_0 = p[:, 0]
    
    #Find the length of the trajectory
    N = p.shape[1]
    
    #Create very boring orientation odometry (constant orientation)
    q_0 = np.array([[1], [0], [0], [0]])
    q = np.zeros((4, N))
    q[:, 0] = q_0.flatten()
    for t in range(1, N):
        q[:, t] = q[:, t - 1]
    
    return p, p_0, q, q_0, N

def quat2Rot(q):
    # Convert quaternion (with scalar first) to a rotation matrix.
    # q is a 4-element array: [w, x, y, z]
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(y**2+z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2+z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2+y**2)]
    ])
    return R


def quatprod(q, r):
    # Quaternion product of q and r.
    # q and r are expected to be 1D numpy arrays of length 4 in the order [q0, q1, q2, q3]
    w = q[0] * r[0] - np.dot(q[1:], r[1:])
    xyz = q[0] * r[1:] + r[0] * q[1:] + np.cross(q[1:], r[1:])
    return np.concatenate(([w], xyz))

def quat2angleaxis(q):
    # Convert quaternion to angle-axis representation.
    # q is a 1D numpy array of length 4 in the order [q0, q1, q2, q3]
    # Angle is 2*acos(q0) and the axis is the normalized vector part times the angle.
    eps = 1e-12
    # Ensure the quaternion is normalized to avoid numerical issues.
    norm_q = np.linalg.norm(q)
    if norm_q == 0:
        # Avoid division by zero; return zero vector if invalid quaternion.
        return np.zeros(3)
    q = q / norm_q
    angle = 2 * np.arccos(q[0])
    sin_term = np.sqrt(1 - q[0]**2)
    if sin_term < eps:
        # If the angle is very small, return a zero vector
        return np.zeros(3)
    axis = q[1:] / sin_term
    return axis * angle

def exp_q_L(delta, q):
    # Deduces the orientation DR using the increment in angle-axis (delta) and the previous quaternion q.
    # Implements left multiplication: new_q = quatprod(exp_quat(delta), q)
    theta = np.linalg.norm(delta)
    eps = 1e-12
    if theta < eps:
        dq = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        dq = np.concatenate(([np.cos(theta / 2.0)], np.sin(theta / 2.0) * (delta / theta)))
    return quatprod(dq, q)

def pseudo_odometry(p, q, R_p, R_q, p_0, q_0, T, bias):
    
    # Create noisy odometry measurements of position
    N = p.shape[1]
    # Compute differences in position
    delta_p_diff = np.hstack((p[:, 1:] - p[:, :-1], np.zeros((3, 1))))
    # Add noise sampled from a multivariate normal distribution with covariance R_p
    noise_p = np.random.multivariate_normal(np.zeros(3), R_p, N).T
    delta_p = delta_p_diff + noise_p + bias.reshape(3, 1)

    # Deduce some orientation odometry
    delta_q = np.zeros((3, N))
    for t in range(N - 1):
        # q_t_C = [-q(1,t); q(2:4,t)]
        q_t_C = np.concatenate(([-q[0, t]], q[1:4, t].reshape(3)))
        # quatprod(q(:,t+1), q_t_C)
        quat_mult = quatprod(q[:, t+1], q_t_C)
        # Convert quaternion to angle-axis and add noise
        noise_q = np.random.multivariate_normal(np.zeros(3), R_q)
        delta_q[:, t] = quat2angleaxis(quat_mult) + noise_q

    # Deduce the position and orientation DR
    p_DR = np.zeros((3, N))
    q_DR = np.zeros((4, N))
    p_DR[:, 0] = p_0.reshape(3)
    q_DR[:, 0] = q_0.reshape(4)

    for t in range(1, N):
        p_DR[:, t] = p_DR[:, t - 1] + delta_p[:, t - 1]
        q_DR[:, t] = exp_q_L(delta_q[:, t - 1], q_DR[:, t - 1])
    
    return delta_p, delta_q, p_DR, q_DR, N


def quatprod(q1, q2):
    # Compute the quaternion product q = q1 * q2.
    # Both q1 and q2 are arrays of shape (4,) with q[0] as scalar.
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat2angleaxis(q):
    # Convert a quaternion to angle-axis representation.
    # Returns a 3-vector: angle * axis.
    # q is assumed to be [w, x, y, z] with w as scalar.
    if q[0] > 1.0:
        q = q / np.linalg.norm(q)
    angle = 2 * np.arccos(q[0])
    s = np.sqrt(1 - q[0]**2)
    if s < 1e-8:
        # If s is close to zero, direction is not important
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = q[1:4] / s
    return angle * axis

def EKF_UKF_localization(N, delta_p, delta_q, y_mag, q_0, p_0, P_0, p, R_p, R_q, sigma_y, pinn_model):
      
    # n number of state
    n = 6
    alpha = 1
    beta = 1
    kappa = 3 - n

    lambda_val = alpha**2 * (n + kappa) - n        
    # lambda_val = alpha**2        
    WM = np.zeros((2*n+1, 1))
    WC = np.zeros((2*n+1, 1))
    for j in range(2*n+1):
        if j == 0:
            wm = lambda_val / (n + lambda_val)
            wc = lambda_val / (n + lambda_val) + (1 - alpha**2 + beta)
        else:
            wm = 1 / (2 * (n + lambda_val))
            wc = wm
        WM[j, 0] = wm
        WC[j, 0] = wc

    # Pre-allocate position trajectory for all particles
    q_UKF = np.zeros((4, N))
    p_UKF = np.zeros((3, N))
    # Pre-allocate space for the position covariance
    P_EKF = P_0.copy()
    # P_UKF = 0.001*eye(7);
    P_UKF = 0.001 * np.eye(6)
    # Q_UKF = [R_p, zeros(3,4); zeros(4,3), 0.0001*eye(4)*T];
    Q_UKF = np.block([[R_p, np.zeros((3, 3))],
                        [np.zeros((3, 3)), R_q]])
    R_UKF = sigma_y**2 * np.eye(3)
    # Initialise orientation and position estimates
    q_UKF[:, 0] = q_0.reshape((4,))
    p_UKF[:, 0] = p_0.reshape((3,))

    # mean for UKF
    # q_0_u = q_0(1:3); %% rename to r (rotation not quaternion)
    r_0 = quat2eul(q_0)
    m_state = np.hstack((p_0, r_0))  # convert q to r (Euler angles)


    # Iterate through all timesteps
    # for t in range(1, 2):
    for t in range(1, N):
        # start implementing UKF
        # Form the sigma points for dynamic model
        A = cholesky(P_UKF, lower=True)
        SX = np.hstack((np.zeros((n, 1)), A, -A))
        SX = np.sqrt(n + lambda_val) * SX + np.tile(m_state.reshape(-1, 1), (1, SX.shape[1]))
        
        # Propagate through the dynamic model
        SXp = SX[0:3, :].copy()  # at time t-1
        SXr = SX[3:, :].copy()   # at time t-1

        for i_sigma in range(2*n+1):
            SXp[:, i_sigma] = SXp[:, i_sigma] + delta_p[:, t-1]
            a = eul2quat(SXr[:, i_sigma].reshape(3, 1))
            b = exp_q_L(delta_q[:, t-1], a)
            b_eul = quat2eul(b)
            SXr[:, i_sigma] = b_eul
        
        HX = np.vstack((SXp, SXr))
        
        # Compute the predicted mean and covariance
        m_state = np.zeros((m_state.shape[0],))
        P_UKF = np.zeros(P_UKF.shape)
        for i_sigma in range(HX.shape[1]):
            m_state = m_state + WM[i_sigma, 0] * HX[:, i_sigma]
        for i_sigma in range(HX.shape[1]):
            diff = (HX[:, i_sigma] - m_state).reshape(-1, 1)
            P_UKF = P_UKF + WC[i_sigma, 0] * (diff * diff.T)
        P_UKF = P_UKF + Q_UKF
        # Form sigma points for measurement step and propagate through the measurement model
        A = cholesky(P_UKF, lower=True)
        SX_m = np.hstack((np.zeros((m_state.shape[0], 1)), A, -A))
        SX_m = np.sqrt(n + lambda_val) * SX_m + np.tile(m_state.reshape(-1, 1), (1, SX_m.shape[1]))
        
        SXp = SX_m[0:3, :].copy()  # at time t-1
        SXr = SX_m[3:, :].copy()   # at time t-1

        HY = np.zeros((3, 2*n+1))
        for i_sigma in range(2*n+1):
            
            # pinn_model = tf.keras.models.load_model(model_path)
            x_test = tf.fill([1,1], float(SXp[0, i_sigma]))
            y_test = tf.fill([1,1], float(SXp[1, i_sigma]))
            z_test = tf.fill([1,1], float(SXp[2, i_sigma]))
 
            inputs = tf.concat([x_test, y_test, z_test], axis = 1)                
            # mage_value = tf.reshape(pinn_model.call(inputs), [3,1])
            # pinn_model = tf.keras.models.load_model(model_path)
            HY[:, i_sigma] = pinn_model.call(inputs)
        
        # Compute the updated mean and covariance
        mu = np.zeros((HY.shape[0], 1))
        S = np.zeros((HY.shape[0], HY.shape[0]))
        C = np.zeros((SX.shape[0], HY.shape[0]))
        for i_sigma in range(SX.shape[1]):
            mu = mu + WM[i_sigma, 0] * HY[:, i_sigma].reshape(-1, 1)
        for i_sigma in range(SX.shape[1]):
            diff_mu = (HY[:, i_sigma].reshape(-1, 1) - mu)
            S = S + WC[i_sigma, 0] * (diff_mu @ diff_mu.T)
            diff_cross = (SX[:, i_sigma].reshape(-1, 1) - m_state.reshape(-1, 1)) * (HY[:, i_sigma].reshape(1, -1) - mu.T)
            C = C + WC[i_sigma, 0] * diff_cross
        S = S + R_UKF

        # Compute the gain and updated mean and covariance  
        # K = C * inv(S)
        K = np.matmul(C, inv(S))
        # m_state = m_state.reshape(-1, 1) + K * (y_mag[:, t].reshape(-1, 1) - mu)
        m_state = m_state.reshape(-1, 1) + np.matmul(K, (y_mag[:, t].reshape(-1, 1) - mu))
        m_state = m_state.flatten()
        
        ks = np.matmul(K, S)
        dpekf = np.matmul(ks, K.T)
        P_EKF = P_EKF - dpekf
        # P_UKF = P_UKF - K * S * K.T
        P_UKF = 0.5 * (P_UKF + P_UKF.T)
        
        p_UKF[:, t] = m_state[0:3]
        q_UKF[:, t] = eul2quat(m_state[3:].reshape(3, 1))
            
    return p_UKF, q_UKF



# ------------------------ Main Simulation Code ------------------------
if __name__ == '__main__':
    print('Start simulations')
    tic = time.time()
    model_path = "saved_models/Circular_loops_4x64_N_u_90"
    pinn_model = tf.keras.models.load_model(model_path)


    seed = 42
    np.random.seed(seed)

    # Prepare folder for saving results in
    current_time = time.localtime()
    time_vec = [current_time.tm_year, current_time.tm_mon, current_time.tm_mday,
                current_time.tm_hour, current_time.tm_min, current_time.tm_sec]
    # Using time.strftime to mimic Matlab's date function (format: dd-mmm-yyyy)
    date_str = time.strftime("%d-%b-%Y", current_time)
    foldername = 'LocalisationMCResultsLengthscales/Run' + date_str + '-' + str(time_vec[3]) + '-' + str(time_vec[4])

    experiments = 100
    

    # Set room dimensions
    margin = 0.5
    xl = -1.5
    xu = 1.5
    yl = -1.5
    yu = 1.5
    zl = -0.5
    zu = 0.5
    T = 0.1

    # Simulate trajectory instead
    z = 0.0
    v = 0.5
    gap = 0.1
    p, p_0_gt, q, q_0, N = rectangular_walk_ground_truth(T, xl, xu, yl, yu, margin, z, v)
    p_EKF_M = np.zeros((3, N, experiments))
    p_UKF_M = np.zeros((3, N, experiments))
    # Decide upon the number of basis functions that causes the GP
    # approximation error of 100 samples to be less than the measurement noise
    # Magnetic field params
    sigma_SE = 0.1
    sigma_y = 0.03
    sigma_lin = 0
    l_SE_dummy = 0.3


    p_UKFs = []
    q_UKFs = []
    error_UKFs=[]
    error_DRs=[]
    q_error_UKFs=[]
    q_error_DRs=[]
    
    initial_error = 0.1
    for experiment in range(1, experiments + 1):
        # Simulate magnetic field measurements
        y_mag = np.zeros((3, N))
        for t in range(N):
        # for t in range(10):
            # pinn_model = tf.keras.models.load_model(model_path)
            x_test = tf.fill([1,1], float(p[0, t]))
            y_test = tf.fill([1,1], float(p[1, t]))
            z_test = tf.fill([1,1], float(p[2, t]))

            inputs = tf.concat([x_test, y_test, z_test], axis = 1)                
            mage_value = tf.reshape(pinn_model.call(inputs), [3,1])
            mage_real = np.array([B(x_test, y_test, y_test)]).reshape(3,1)
            
            res = (mage_value - mage_real).numpy()

            # Get rotation matrix from quaternion q[:,t] and transpose it.
            Rot = quat2Rot(q[:, t])
            # Add measurement noise
            noise = np.random.multivariate_normal(np.zeros(3), (sigma_y**2)*np.eye(3))
            noise = noise.reshape([3,1])
            xxx = np.dot(Rot.T, mage_value) + noise
            y_mag[:, t] = (np.dot(Rot, mage_value) + noise).reshape((3,))

        # Number of particles used in visuals

        # Noise parameters
        R_p = 0.0001 * np.eye(3) * T
        R_q = 0.001 * np.eye(3) * T

        # Init covariances
        P_0_cov = 0.001 * np.eye(6)
        P_0_cov[0:2, 0:2] = initial_error * np.eye(2)

        # Add error to the position estimate
        p_0 = p_0_gt + np.array([np.sqrt(0.5)*initial_error, np.sqrt(0.5)*initial_error, 0])

        # Find odometry measurements
        bias = np.array([0, 0, 0])
        delta_p, delta_q, p_DR, q_DR, N_val = pseudo_odometry(p, q, R_p, R_q, p_0, q_0, T, bias)

        # Final plots for result presentation
        fontsize = 9

        # Run filter with higher measurement error to better visualise the
        # influence of the measurement on the PF distribution
        sigma_y = 0.3

        # Run the filters
        p_UKF, q_UKF = EKF_UKF_localization(N, delta_p, delta_q, y_mag, q_0, p_0, P_0_cov, p, R_p, R_q, sigma_y, pinn_model)
        
        p_error_UKF= p - p_UKF  # shape (3, N)
        q_error_UKF= q - q_UKF  # shape (3, N)
        p_error_DR = p - p_DR  # shape (3, N)
        q_error_DR= q - q_DR  # shape (3, N)
        
        
        error_UKFs.append(p_error_UKF)
        error_DRs.append(p_error_DR)
        


        print('experiment number: ' + str(experiment) + '/' + str(experiments))


    ##Final Plots
    # Convert list of RMSEs to array of shape (experiments, N)
    error_UKFs = np.array(error_UKFs)  # shape (experiments, N)
    error_DRs = np.array(error_DRs)


    rmse_UKF = np.sqrt(np.sum(np.square(error_UKFs), axis=(0,1))/experiments)
    # std_rmse_UKF = np.std(rmse_array_UKF, axis=0)
    rmse_DR = np.sqrt(np.sum(np.square(error_DRs), axis=(0,1))/experiments)
    

   
# End of simulation

plt.plot(range(N), rmse_UKF, color='blue', linewidth=1.5, label='UKF RMSE')
plt.plot(range(N), rmse_DR, color='green', linewidth=1.5, label='Odometry RMSE')

# Labels and title
plt.xlabel('$t$ (s)')
plt.ylabel('RMSE (m)')
plt.title('Poisition RMSE Across 100 Experiments')
plt.legend()
plt.grid(True)
plt.ylim(0,0.3)
plt.tight_layout()
plt.show()


plt.figure()
plt.subplot(2,1,1)
plt.plot(p_UKF[0, :], linewidth=1.4)
plt.plot(p[0, :], linewidth=1.4)
plt.title('$x$ and $y$ States Motion Visualization')
plt.xlabel('$t$(s)')
plt.ylabel('$x$(m)')
plt.legend(['$x$-UKF', '$x$-True'])
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(p_UKF[1, :], linewidth=1.4)
plt.plot(p[1, :], linewidth=1.4)
plt.xlabel('$t$(s)')
plt.ylabel('$y$(m)')
plt.legend(['$y$-UKF', '$y$-True'])
plt.grid(True)

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.square(p_UKF[0, :] - p[0, :]), linewidth=1.4)
plt.title('Visualization of the MSE for the $x$ and $y$ states')
plt.xlabel('$t$(s)')
plt.ylabel('MSE')
plt.legend(['x-state'])
plt.ylim(0, 0.04)
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(np.square(p_UKF[1, :] - p[1, :]), linewidth=1.4)
plt.xlabel('$t$(s)')
plt.ylabel('MSE')
plt.legend(['y-state'])
plt.ylim(0, 0.1)
plt.grid(True)

plt.show()
print("Simulation completed.")
timespent = time.time() - tic
savemat('timespent.mat', {'timespent': timespent})
# # Save the workspace variables in a .mat file. For simplicity, we save a dictionary with key 'Workspace'
workspace = {'p': p, 'q': q, "p_UKF": p_UKF,
            'p_DR': p_DR, 'q_UKF': q_UKF, "q_DR": q_DR, 'y_mag': y_mag,
        'sigma_y': sigma_y, 'R_p': R_p, 'R_q': R_q, 'P_0_cov': P_0_cov, 'p_0': p_0,
        'q_0': q_0, 'N': N, 'experiments': experiments, 'p_0_gt': p_0_gt, 
        'q_0_gt': q_0, 'N_val': N_val,'p': p, 'q': q, 'p_DR': p_DR, 'q_DR': q_DR, 'delta_p': delta_p,
        'delta_q': delta_q,}
savemat(('Workspace.mat'), workspace)
    
    
    
# %%

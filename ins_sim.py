import numpy as np
import matplotlib.pyplot as plt

"""
Strapdown Inertial Navigation System (INS) Simulation
Author: Roey Turjeman
"""

# --- 1. Quaternion Utilities ---
def quat_mult(q, p):
    """
    Multiplies two quaternions.
    q, p: Quaternions defined as [q_w, q_x, q_y, q_z]
    """
    qw, qx, qy, qz = q
    pw, px, py, pz = p
    return np.array([
        qw*pw - qx*px - qy*py - qz*pz,
        qw*px + qx*pw + qy*pz - qz*py,
        qw*py - qx*pz + qy*pw + qz*px,
        qw*pz + qx*py - qy*px + qz*pw
    ])

def quat_to_dcm(q):
    """
    Converts a quaternion [w, x, y, z] to a 3x3 Direction Cosine Matrix (DCM).
    """
    qw, qx, qy, qz = q
    return np.array([
        [qw**2 + qx**2 - qy**2 - qz**2,  2*(qx*qy - qw*qz),              2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),              qw**2 - qx**2 + qy**2 - qz**2,  2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),              2*(qy*qz + qw*qx),              qw**2 - qx**2 - qy**2 + qz**2]
    ])

def quat_normalize(q):
    """Normalizes the quaternion to prevent numerical drift leading to non-unit quaternions."""
    return q / np.linalg.norm(q)

def quat_kinematic_derivative(q, omega):
    """
    Calculates the derivative of the quaternion q_dot = 0.5 * q x [0, omega]
    omega: Angular velocity vector [wx, wy, wz] measured by gyro.
    """
    omega_quat = np.array([0, omega[0], omega[1], omega[2]])
    return 0.5 * quat_mult(q, omega_quat)

# --- 2. Simulation Setup & Constants ---
dt = 0.01  # Time step [s]
total_time = 60.0  # Total simulation time [s]
steps = int(total_time / dt)
time_vec = np.linspace(0, total_time, steps)

# Constants
g = np.array([0, 0, 9.81])  # Gravity vector in Navigation frame (NED/Local Level)

# Define True Motion Profile: Constant acceleration and slight rotation
true_accel_n = np.array([0.5, 0.2, 0.0]) # True acceleration in Nav Frame [m/s^2]
true_omega_b = np.array([0.01, -0.005, 0.02]) # True rotation rate in Body Frame [rad/s]

# --- 3. Sensor Models (Injecting Errors) ---
# Error Model parameters
accel_bias = np.array([0.005, -0.002, 0.01]) # Constant bias in m/s^2
gyro_bias = np.array([0.0001, -0.0002, 0.0001]) # Constant bias in rad/s
# For a deeper interview, you might add White Noise: np.random.normal(0, std_dev)

# --- 4. Simulation Initialization ---
# True states
pos_true = np.zeros((steps, 3))
vel_true = np.zeros((steps, 3))
q_true = np.array([1.0, 0.0, 0.0, 0.0]) # Initially aligned (Nav frame = Body frame)

# INS Computed states
pos_ins = np.zeros((steps, 3))
vel_ins = np.zeros((steps, 3))
q_ins = np.array([1.0, 0.0, 0.0, 0.0])

# Lists for plotting
err_pos = np.zeros((steps, 3))

# --- 5. Main Strapdown Loop ---
for i in range(1, steps):
    # ---------------------------------------------------------
    # A. Propagate TRUE Kinematics (Ground Truth)
    # ---------------------------------------------------------
    # Update True Orientation (q_true)
    dq_true = quat_kinematic_derivative(q_true, true_omega_b) * dt
    q_true = quat_normalize(q_true + dq_true)
    C_b2n_true = quat_to_dcm(q_true)
    C_n2b_true = C_b2n_true.T
    
    # Propagate True Velocity and Position
    vel_true[i] = vel_true[i-1] + true_accel_n * dt
    pos_true[i] = pos_true[i-1] + vel_true[i] * dt
    
    # Create True IMU Measurements
    # Specific force measured by Accelerometer: f^b = C_n2b * (a^n - g^n)
    # Since gravity is acting "downwards" (Z-axis), the IMU feels +g upwards
    true_f_b = C_n2b_true @ (true_accel_n - g)
    
    # ---------------------------------------------------------
    # B. Generate Sensor Measurements (Add Bias)
    # ---------------------------------------------------------
    meas_accel = true_f_b + accel_bias
    meas_gyro = true_omega_b + gyro_bias
    
    # ---------------------------------------------------------
    # C. Execute INS Strapdown Algorithm (Navigation Processor)
    # ---------------------------------------------------------
    
    # 1. Update Attitude (Quaternion Integration) using Gyro Data
    dq_ins = quat_kinematic_derivative(q_ins, meas_gyro) * dt
    q_ins = quat_normalize(q_ins + dq_ins)
    
    # 2. Transform Specific Force to Navigation Frame
    C_b2n_ins = quat_to_dcm(q_ins)
    f_n_ins = C_b2n_ins @ meas_accel
    
    # 3. Calculate Kinematic Acceleration in Nav Frame (remove gravity effect)
    accel_n_ins = f_n_ins + g # +g because f_n_ins contains the -g component measured by IMU
    
    # 4. Integrate to Velocity and Position
    vel_ins[i] = vel_ins[i-1] + accel_n_ins * dt
    pos_ins[i] = pos_ins[i-1] + vel_ins[i] * dt
    
    # Calculate errors for tracking
    err_pos[i] = pos_ins[i] - pos_true[i]

# --- 6. Plotting the Results ---
plt.figure(figsize=(12, 8))

# Subplot 1: True vs INS Computed Position
plt.subplot(2, 2, 1)
plt.plot(pos_true[:, 0], pos_true[:, 1], label='True Trajectory', color='green')
plt.plot(pos_ins[:, 0], pos_ins[:, 1], label='INS Trajectory', color='red', linestyle='dashed')
plt.title('Position (North vs East)')
plt.xlabel('North Position (m)')
plt.ylabel('East Position (m)')
plt.legend()
plt.grid(True)

# Subplot 2: Position Error Over Time (Drift)
plt.subplot(2, 2, 2)
plt.plot(time_vec, err_pos[:, 0], label='Error North (x)')
plt.plot(time_vec, err_pos[:, 1], label='Error East (y)')
plt.plot(time_vec, err_pos[:, 2], label='Error Down (z)')
plt.title('Position Error (Drift) Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.legend()
plt.grid(True)

# Subplot 3: Velocity True vs INS (North)
plt.subplot(2, 2, 3)
plt.plot(time_vec, vel_true[:, 0], label='True Vel North', color='blue')
plt.plot(time_vec, vel_ins[:, 0], label='INS Vel North', color='orange', linestyle='dashed')
plt.title('Velocity (North Axis)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Simulation complete. Notice how the Position Error grows non-linearly over time due to double integration of uncompensated bias.")

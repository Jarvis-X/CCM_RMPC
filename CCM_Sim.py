import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def plot_rccm_results(pos, ref, t, err_norm, tube_alpha=None, dist_max=1.5):
    """
    Visualizes the simulation results.
    
    Args:
        pos: (N, 3) Actual position history
        ref: (N, 3) Nominal position history
        t:   (N,) Time vector
        err_norm: (N,) Norm of the error vector state
        tube_alpha: (Optional) The theoretical alpha value to draw the bound line
        dist_max: (Optional) The magnitude of disturbance used in sim (for bound calc)
    """
    # Use seaborn style if available for nicer plots, else default
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('grid')

    fig = plt.figure(figsize=(14, 10))

    # ==========================================
    # 1. 3D TRAJECTORY PLOT (Like Fig 5 in paper)
    # ==========================================
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Plot Reference
    ax1.plot(ref[:,0], ref[:,1], ref[:,2], 'k--', linewidth=1.5, label='Nominal Trajectory', alpha=0.7)
    
    # Plot Actual
    ax1.plot(pos[:,0], pos[:,1], pos[:,2], 'b-', linewidth=2, label='Actual (RCCM)')
    
    # Mark Start and End
    ax1.scatter(pos[0,0], pos[0,1], pos[0,2], c='g', marker='o', s=50, label='Start')
    ax1.scatter(pos[-1,0], pos[-1,1], pos[-1,2], c='r', marker='x', s=50, label='End')
    
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_title('3D Flight Path')
    ax1.legend()
    ax1.view_init(elev=30, azim=-45)

    # ==========================================
    # 2. POSITION COMPONENTS VS TIME
    # ==========================================
    ax2 = fig.add_subplot(2, 2, 2)
    labels = ['x', 'y', 'z']
    colors = ['r', 'g', 'b']
    
    for i in range(3):
        ax2.plot(t, pos[:,i], color=colors[i], label=f'Actual {labels[i]}')
        ax2.plot(t, ref[:,i], color=colors[i], linestyle='--', alpha=0.5)
    
    # Highlight Disturbance Region
    ax2.axvspan(2.0, 4.0, color='yellow', alpha=0.2, label='Wind Disturbance')
    
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Position [m]')
    ax2.set_title('Tracking Performance (XYZ)')
    ax2.legend(loc='upper right', fontsize='small')

    # ==========================================
    # 3. ERROR NORM (THE TUBE) (Like Fig 6 in paper)
    # ==========================================
    ax3 = fig.add_subplot(2, 1, 2)
    
    ax3.plot(t, err_norm, 'k-', linewidth=2, label='||Error State||')
    
    # Draw Theoretical Tube Bound if provided
    if tube_alpha is not None:
        theoretical_bound = tube_alpha * dist_max
        ax3.axhline(y=theoretical_bound, color='r', linestyle='--', linewidth=2, label=f'Theoretical Tube (alpha={tube_alpha:.2f})')
        
        # Shade the safety region
        ax3.fill_between(t, 0, theoretical_bound, color='green', alpha=0.1)
        ax3.text(t[0], theoretical_bound + 0.05, "Safety Guarantee", color='r')

    # Highlight Disturbance Region again
    ax3.axvspan(2.0, 4.0, color='yellow', alpha=0.2, label='Disturbance Active')

    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Error Norm')
    ax3.set_title('Tube Compliance: Does error stay bounded?')
    ax3.legend()
    ax3.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

class RCCMSimulator:
    def __init__(self, W_coeffs, Y_coeffs, basis_exps):
        self.Ws = W_coeffs
        self.Ys = Y_coeffs
        self.exps = basis_exps
        
        self.m = 1.0
        self.g = 9.81
        self.J = np.diag([0.02, 0.02, 0.04])
        self.J_inv = np.linalg.inv(self.J)
        
    def get_controller_gain_quat(self, quat):
        """
        Reconstructs K(q) numerically given the current quaternion.
        Args:
            quat: np.array [qw, qx, qy, qz] (normalized)
        """
        # 1. Evaluate Basis Monomials
        # self.exps contains tuples like (2, 0, 1, 1) for qw^2 * qy * qz
        weights = []
        for exp in self.exps:
            val = 1.0
            # Iterate through the 4 quaternion components
            for i, p in enumerate(exp):
                if p > 0:
                    val *= (quat[i] ** p)
            weights.append(val)
            
        # 2. Reconstruct W and Y matrices via tensor sum
        # W_num = sum( coeff_k * weight_k )
        # We use a loop or list comprehension (fast enough for ~70 basis terms)
        W_num = sum(self.Ws[k] * w for k, w in enumerate(weights))
        Y_num = sum(self.Ys[k] * w for k, w in enumerate(weights))
        
        # 3. Regularization (Crucial for high-degree polynomials)
        # Prevents numerical singularity if the polynomial dips slightly
        W_reg = W_num + 1e-6 * np.eye(12)
        
        # 4. Compute K = Y * inv(W)
        # Using solve(A, B) for A*X = B => X = A\B
        # We want K = Y @ inv(W) => K.T = inv(W).T @ Y.T = inv(W) @ Y.T
        try:
            K_num = np.linalg.solve(W_reg, Y_num.T).T
        except np.linalg.LinAlgError:
            print("WARNING: W(q) is singular. Returning zero gain.")
            return np.zeros((6, 12))
            
        return K_num
        
    def get_controller_gain(self, R_curr):
        r_flat = R_curr.flatten()
        weights = []
        for exp in self.exps:
            val = 1.0
            for i, p in enumerate(exp):
                if p > 0: val *= (r_flat[i]**p)
            weights.append(val)
            
        # Reconstruct W and Y
        W_num = sum(self.Ws[k] * w for k, w in enumerate(weights))
        Y_num = sum(self.Ys[k] * w for k, w in enumerate(weights))
        
        # --- FIX 1: REGULARIZATION ---
        # A Degree 4 polynomial metric can become ill-conditioned between sample points.
        # We add a tiny epsilon to the diagonal to ensure invertibility.
        W_reg = W_num + 1e-6 * np.eye(12)
        
        # Check Condition Number (Optional debug)
        # if np.linalg.cond(W_reg) > 1e5: print("Warning: W is ill-conditioned")

        try:
            K_num = np.linalg.solve(W_reg, Y_num.T).T
        except np.linalg.LinAlgError:
            print("WARNING: W(R) is singular. Returning zero gain.")
            return np.zeros((6, 12)) # Safety fallback
            
        return K_num

    def nominal_trajectory(self, t):
        # -------------------------------------------------
        # 1. TRANSLATIONAL (Figure-8)
        # -------------------------------------------------
        omega_traj = 1.0
        
        # Position
        px = np.sin(omega_traj * t)
        py = np.sin(omega_traj * t / 2)
        pz = 1.0 
        
        # Velocity
        vx = omega_traj * np.cos(omega_traj * t)
        vy = (omega_traj/2) * np.cos(omega_traj * t / 2)
        vz = 0
        
        # Acceleration
        ax = -(omega_traj**2) * np.sin(omega_traj * t)
        ay = -(omega_traj/2)**2 * np.sin(omega_traj * t / 2)
        az = 0
        
        # Feedforward Force
        g_vec = np.array([0, 0, -self.g])
        f_nom = self.m * (np.array([ax, ay, az]) - g_vec)

        # -------------------------------------------------
        # 2. ROTATIONAL (Yaw Spin + Pitch Nod)
        # -------------------------------------------------
        psi = 0.5 * t              # Yaw
        theta = 0.35 * np.sin(2*t) # Pitch
        phi = 0.0                  # Roll
        
        # Derivatives
        dpsi = 0.5
        dtheta = 0.35 * 2 * np.cos(2*t)
        dphi = 0.0
        
        ddpsi = 0.0
        ddtheta = -0.35 * 4 * np.sin(2*t)
        ddphi = 0.0
        
        # A. Reference Rotation Matrix (The Fix!)
        # Create Rotation object from Euler angles
        R_obj = Rotation.from_euler('zyx', [psi, theta, phi])
        R_ref = R_obj.as_matrix() # Returns 3x3 Matrix (Size 9 when flattened)
        
        # B. Reference Angular Velocity (Body Frame)
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        
        wx = dphi - dpsi * st
        wy = dtheta * cp + dpsi * ct * sp
        wz = dpsi * ct * cp - dtheta * sp
        w_ref = np.array([wx, wy, wz])
        
        # C. Reference Angular Acceleration
        dwx = ddphi - (ddpsi * st + dpsi * ct * dtheta)
        dwy = (ddtheta*cp - dtheta*sp*dphi) + \
              (ddpsi*ct*sp - dpsi*st*dtheta*sp + dpsi*ct*cp*dphi)
        dwz = (ddpsi*ct*cp - dpsi*st*dtheta*cp - dpsi*ct*sp*dphi) - \
              (ddtheta*sp + dtheta*cp*dphi)
              
        dw_ref = np.array([dwx, dwy, dwz])
        
        # Feedforward Torque
        tau_nom = self.J @ dw_ref + np.cross(w_ref, self.J @ w_ref)
        
        # Combine Inputs
        u_nom = np.hstack([f_nom, tau_nom])
        
        return np.hstack([px, py, pz]), R_ref, np.hstack([vx, vy, vz]), w_ref, u_nom
    
    def dynamics(self, t, x_state, u_control, w_dist):
        # (Same as before)
        R_flat = x_state[3:12]
        R = R_flat.reshape((3,3))
        v = x_state[12:15]
        w = x_state[15:18]
        
        f_W = u_control[0:3]
        tau_B = u_control[3:6]
        
        # Clamp inputs to realistic motor limits (Crucial for sim stability)
        # e.g., Max Force 20N, Max Torque 2Nm
        f_W = np.clip(f_W, -50, 50) 
        tau_B = np.clip(tau_B, -10, 10)
        
        f_dist = w_dist[0:3]
        tau_dist = w_dist[3:6]
        
        dp = v
        w_hat = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        dR = R @ w_hat
        
        g_vec = np.array([0, 0, -self.g])
        dv = (1.0/self.m) * (f_W + f_dist) + g_vec
        
        torque_net = tau_B + tau_dist - np.cross(w, self.J @ w)
        dw = self.J_inv @ torque_net
        
        return np.concatenate([dp, dR.flatten(), dv, dw])

    def rk4_step(self, t, state, u, dist, dt):
        """ Runge-Kutta 4 Integration Step """
        k1 = self.dynamics(t, state, u, dist)
        k2 = self.dynamics(t + dt/2, state + dt/2 * k1, u, dist)
        k3 = self.dynamics(t + dt/2, state + dt/2 * k2, u, dist)
        k4 = self.dynamics(t + dt, state + dt * k3, u, dist)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def run_simulation(self, duration=10.0, dt=0.001, rot_representation='R'):
        steps = int(duration / dt)
        times = np.linspace(0, duration, steps)
        
        p0, R0, v0, w0, _ = self.nominal_trajectory(0)
        p_act = p0 + np.array([0.1, -0.1, 0.05]) 
        R_act = R0
        state = np.concatenate([p_act, R_act.flatten(), v0, w0])
        
        pos_hist = []
        pos_ref_hist = []
        err_norm_hist = []
        
        print(f">>> Starting RK4 Simulation (dt={dt}s)...")
        
        for i, t in enumerate(times):
            p_ref, R_ref, v_ref, w_ref, u_nom = self.nominal_trajectory(t)
            
            p_curr = state[0:3]
            R_curr = state[3:12].reshape((3,3))
            v_curr = state[12:15]
            w_curr = state[15:18]
            
            e_p = p_curr - p_ref
            e_v = v_curr - v_ref
            e_w = w_curr - w_ref
            
            R_err = R_ref.T @ R_curr
            skew_sym = 0.5 * (R_err - R_err.T)
            eta = np.array([skew_sym[2,1], skew_sym[0,2], skew_sym[1,0]])
            
            error_vec = np.concatenate([e_p, eta, e_v, e_w])
            
            # Control
            if rot_representation == 'R':
                K = self.get_controller_gain(R_curr)
            elif rot_representation == 'quat':
                quat = Rotation.from_matrix(R_curr).as_quat()  # [x, y, z, w]
                K = self.get_controller_gain(quat[[3,0,1,2]])  # Convert to [w, x, y, z]
            u_fb = K @ error_vec 
            u_total = u_nom + u_fb
            
            # Disturbance
            dist = np.random.uniform(0.0, 1.0, size=6)
            if 2.0 < t < 4.0:
                dist[0] += 1.5 
                dist[3] += 0.1 
            
            # --- FIX 2: RK4 INTEGRATION ---
            state = self.rk4_step(t, state, u_total, dist, dt)
            
            # Divergence Check
            if np.linalg.norm(state[0:3]) > 100.0:
                print(f"!!! SIMULATION DIVERGED at t={t:.2f}s !!!")
                break

            # Re-orthonormalize R
            U, _, Vt = np.linalg.svd(state[3:12].reshape((3,3)))
            state[3:12] = (U @ Vt).flatten()
            
            pos_hist.append(p_curr)
            pos_ref_hist.append(p_ref)
            err_norm_hist.append(np.linalg.norm(error_vec))

        return np.array(pos_hist), np.array(pos_ref_hist), times[:len(pos_hist)], np.array(err_norm_hist)
    
if __name__ == "__main__":
    # 1. Load the data
    print("Loading controller data...")
    with open("rccm_controller_data.pkl", "rb") as f:
        data = pickle.load(f)

    # 2. Unpack
    W_vals = data["W_matrices"]
    Y_vals = data["Y_matrices"]
    basis_exps = data["basis_exponents"]
    tube_size = data["alpha"]

    print(f"Loaded Controller. Tube Size alpha = {tube_size:.4f}")

    # 3. Initialize Simulator
    sim = RCCMSimulator(W_vals, Y_vals, basis_exps)

    # 4. Run
    pos, ref, t, err = sim.run_simulation(rot_representation='R')
    alpha_val = data["alpha"] # from pickle load
    plot_rccm_results(pos, ref, t, err, tube_alpha=alpha_val, dist_max=1.5)
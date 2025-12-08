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
        """
        W_coeffs: List of numerical matrices [W_0, W_1, ...]
        Y_coeffs: List of numerical matrices [Y_0, Y_1, ...]
        basis_exps: List of exponents tuples corresponding to the coeffs
        """
        self.Ws = W_coeffs
        self.Ys = Y_coeffs
        self.exps = basis_exps
        
        # System Constants
        self.m = 1.0
        self.g = 9.81
        self.J = np.diag([0.02, 0.02, 0.04])
        self.J_inv = np.linalg.inv(self.J)
        
    def get_controller_gain(self, R_curr):
        """
        Reconstructs K(R) numerically for the current orientation.
        Fast and robust.
        """
        # 1. Evaluate Basis
        r_flat = R_curr.flatten()
        weights = []
        for exp in self.exps:
            val = 1.0
            for i, p in enumerate(exp):
                if p > 0: val *= (r_flat[i]**p)
            weights.append(val)
            
        # 2. Sum Matrices
        # Using tensordot or simple loops (loop is fine for small basis)
        W_num = sum(w * c for w, c in zip(weights, self.Ws))
        Y_num = sum(y * c for y, c in zip(weights, self.Ys))
        
        # 3. Compute K = Y * inv(W)
        # solve(A, B) solves A*X = B -> X = inv(A)*B? No, solve(A, b) is A*x=b.
        # We want K = Y @ inv(W).
        # This is equivalent to solving W.T @ K.T = Y.T
        K_num = np.linalg.solve(W_num.T, Y_num.T).T
        return K_num

    def nominal_trajectory(self, t):
        """
        Returns reference state and input for a Figure-8 trajectory.
        Full State: [p, R_flat, v, w]
        """
        # Figure 8 Position
        omega_traj = 1.0
        px = np.sin(omega_traj * t)
        py = np.sin(omega_traj * t / 2)
        pz = 1.0 # Hover height
        
        vx = omega_traj * np.cos(omega_traj * t)
        vy = (omega_traj/2) * np.cos(omega_traj * t / 2)
        vz = 0
        
        ax = -(omega_traj**2) * np.sin(omega_traj * t)
        ay = -(omega_traj/2)**2 * np.sin(omega_traj * t / 2)
        az = 0
        
        # Orientation: Keep it simple (Identity) for this test
        # A real planner would align heading with velocity
        R_ref = np.eye(3)
        w_ref = np.zeros(3)
        dw_ref = np.zeros(3)
        
        # Nominal Inputs (Inverse Dynamics)
        # f_W = m * (a - g) -> But dynamics are dot_v = 1/m f_W + g
        # So f_W = m * (dot_v - g)
        g_vec = np.array([0, 0, -self.g])
        acc_vec = np.array([ax, ay, az])
        f_nom = self.m * (acc_vec - g_vec)
        
        # tau_B = J*dw + w x Jw
        tau_nom = self.J @ dw_ref + np.cross(w_ref, self.J @ w_ref)
        
        u_nom = np.hstack([f_nom, tau_nom])
        
        return np.hstack([px, py, pz]), R_ref, np.hstack([vx, vy, vz]), w_ref, u_nom

    def dynamics(self, t, x_state, u_control, w_dist):
        """
        Nonlinear Dynamics of Fully Actuated Multirotor
        x_state: [p(3), R_flat(9), v(3), w(3)] (Size 18)
        """
        # Unpack
        # p = x_state[0:3]
        R_flat = x_state[3:12]
        R = R_flat.reshape((3,3))
        v = x_state[12:15]
        w = x_state[15:18]
        
        f_W = u_control[0:3]
        tau_B = u_control[3:6]
        
        # Disturbances
        f_dist = w_dist[0:3]
        tau_dist = w_dist[3:6]
        
        # Equations of Motion
        # 1. p_dot = v
        dp = v
        
        # 2. R_dot = R * skew(w)
        w_hat = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        dR = R @ w_hat
        
        # 3. v_dot = 1/m * (f_W) + g + 1/m * f_dist
        # Note: If f_W is world force, gravity is just vector addition
        g_vec = np.array([0, 0, -self.g])
        dv = (1.0/self.m) * (f_W + f_dist) + g_vec
        
        # 4. w_dot = J_inv * (tau_B - w x Jw + tau_dist)
        torque_net = tau_B + tau_dist - np.cross(w, self.J @ w)
        dw = self.J_inv @ torque_net
        
        return np.concatenate([dp, dR.flatten(), dv, dw])

    def run_simulation(self, duration=10.0, dt=0.01):
        steps = int(duration / dt)
        times = np.linspace(0, duration, steps)
        
        # Initial State (Slightly perturbed)
        p0, R0, v0, w0, _ = self.nominal_trajectory(0)
        p_act = p0 + np.array([0.2, -0.2, 0.1]) # Position Error
        R_act = R0 # Start with correct orientation
        state = np.concatenate([p_act, R_act.flatten(), v0, w0])
        
        # History
        pos_hist = []
        pos_ref_hist = []
        err_norm_hist = []
        
        print(">>> Starting Simulation...")
        for i, t in enumerate(times):
            # 1. Get Nominal
            p_ref, R_ref, v_ref, w_ref, u_nom = self.nominal_trajectory(t)
            
            # 2. Unpack Current State
            p_curr = state[0:3]
            R_curr = state[3:12].reshape((3,3))
            v_curr = state[12:15]
            w_curr = state[15:18]
            
            # 3. Compute Error State
            # Matches order: [p, eta, v, w]
            e_p = p_curr - p_ref
            e_v = v_curr - v_ref
            e_w = w_curr - w_ref
            
            # Attitude Error (Geodesic/Log map approximation)
            # R_err = R_ref.T @ R_curr
            # eta approx 0.5 * vee(R_err - R_err.T)
            R_err = R_ref.T @ R_curr
            skew_sym = 0.5 * (R_err - R_err.T)
            eta = np.array([skew_sym[2,1], skew_sym[0,2], skew_sym[1,0]])
            # Rotate eta back to world frame? 
            # Our derivation A-matrix assumed eta was tangent to R. 
            # Usually K expects eta in Body frame.
            
            error_vec = np.concatenate([e_p, eta, e_v, e_w])
            
            # 4. Compute Control (RCCM)
            # Get K for current orientation
            K = self.get_controller_gain(R_curr)
            
            # Feedback u_fb = K * error
            # Note: K is 6x12. Error is 12x1.
            u_fb = K @ error_vec # Negative feedback
            
            u_total = u_nom + u_fb
            
            # 5. Disturbance (Wind Gusts)
            dist = np.zeros(6)
            if 2.0 < t < 4.0:
                dist[0] = 1.5 # 1.5 N wind in X
                dist[3] = 0.1 # Small torque disturbance
            
            # 6. Step Dynamics (Euler Integration)
            dstate = self.dynamics(t, state, u_total, dist)
            state = state + dstate * dt
            
            # Re-orthonormalize R (Crucial for integration stability)
            U, _, Vt = np.linalg.svd(state[3:12].reshape((3,3)))
            R_clean = U @ Vt
            state[3:12] = R_clean.flatten()
            
            # Log
            pos_hist.append(p_curr)
            pos_ref_hist.append(p_ref)
            err_norm_hist.append(np.linalg.norm(error_vec))

        return np.array(pos_hist), np.array(pos_ref_hist), times, err_norm_hist

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
    pos, ref, t, err = sim.run_simulation()
    alpha_val = data["alpha"] # from pickle load
    plot_rccm_results(pos, ref, t, err, tube_alpha=alpha_val, dist_max=1.5)
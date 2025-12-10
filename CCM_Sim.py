import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from tqdm import tqdm

# ==========================================
# 1. VISUALIZATION
# ==========================================
def plot_rccm_results(pos, ref, t, err_norm, att_err, u_hist, tube_alpha=None, dist_max=1.5):
    """
    Visualizes simulation results: 3D path, Position Errors, Attitude Errors, Inputs, and Tube.
    Args:
        u_hist: (N, 6) Control inputs [Fx, Fy, Fz, Tx, Ty, Tz]
    """
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except: plt.style.use('grid')

    fig = plt.figure(figsize=(18, 10))

    # --- 1. 3D Flight Path ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(ref[:,0], ref[:,1], ref[:,2], 'k--', lw=1.5, label='Nominal', alpha=0.7)
    ax1.plot(pos[:,0], pos[:,1], pos[:,2], 'b-', lw=2, label='Actual')
    ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]'); ax1.set_zlabel('Z [m]')
    ax1.set_title('3D Flight Path')
    ax1.legend()

    # --- 2. Position Errors (XYZ Separated) ---
    ax2 = fig.add_subplot(2, 3, 2)
    pos_err = pos - ref
    ax2.plot(t, pos_err[:,0], label='Error X')
    ax2.plot(t, pos_err[:,1], label='Error Y')
    ax2.plot(t, pos_err[:,2], label='Error Z')
    ax2.axvspan(2.0, 4.0, color='yellow', alpha=0.2, label='Disturbance')
    ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Error [m]')
    ax2.set_title('Position Tracking Errors')
    ax2.legend(loc='upper right', fontsize='small')

    # --- 3. Attitude Errors (RPY Separated) ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, att_err[:,0], label='Roll Err')
    ax3.plot(t, att_err[:,1], label='Pitch Err')
    ax3.plot(t, att_err[:,2], label='Yaw Err')
    ax3.axvspan(2.0, 4.0, color='yellow', alpha=0.2)
    ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Error [deg]')
    ax3.set_title('Attitude Tracking Errors')
    ax3.legend(loc='upper right', fontsize='small')

    # --- 4. Tube Compliance (Metric) ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, err_norm, 'k-', lw=2, label='||Error State||')
    if tube_alpha:
        bound = tube_alpha * dist_max
        ax4.axhline(bound, color='r', ls='--', label=f'Bound ({tube_alpha:.2f})')
        ax4.fill_between(t, 0, bound, color='green', alpha=0.1)
    ax4.axvspan(2.0, 4.0, color='yellow', alpha=0.2)
    ax4.set_xlabel('Time [s]'); ax4.set_ylabel('Norm')
    ax4.set_title('Tube Compliance Metric')
    ax4.set_ylim(bottom=0)
    ax4.legend()

    # --- 5. Control Forces ---
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, u_hist[:,0], label='Fx')
    ax5.plot(t, u_hist[:,1], label='Fy')
    ax5.plot(t, u_hist[:,2], label='Fz')
    ax5.axvspan(2.0, 4.0, color='yellow', alpha=0.2)
    ax5.set_xlabel('Time [s]'); ax5.set_ylabel('Force [N]')
    ax5.set_title('Control Forces')
    ax5.legend(loc='upper right', fontsize='small')

    # --- 6. Control Torques ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t, u_hist[:,3], label='Tx')
    ax6.plot(t, u_hist[:,4], label='Ty')
    ax6.plot(t, u_hist[:,5], label='Tz')
    ax6.axvspan(2.0, 4.0, color='yellow', alpha=0.2)
    ax6.set_xlabel('Time [s]'); ax6.set_ylabel('Torque [Nm]')
    ax6.set_title('Control Torques')
    ax6.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.savefig("rccm_simulation_results.pdf", dpi=300)
    plt.show()
    
    
# ==========================================
# 2. SIMULATOR CLASS (Auto-Detect Mode)
# ==========================================
import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize, Bounds, LinearConstraint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# ==========================================
# 1. GEODESIC SOLVER (Optimization Engine)
# ==========================================
class FastGeodesicSolver:
    def __init__(self, n, N_nodes, W_func):
        self.n = n
        self.N = N_nodes
        self.W_func = W_func
        
        # Chebyshev Nodes & Weights (Clenshaw-Curtis)
        theta = np.pi * np.arange(0, N_nodes + 1) / N_nodes
        x_cheb = np.cos(theta)
        self.nodes = 0.5 * (1 - x_cheb)
        
        # Precompute Differentiation Matrix D
        # Maps x (values at nodes) -> v (derivatives at nodes)
        # v = x @ D_mat.T
        self.D_mat = self._cheb_diff_matrix(N_nodes)
        
        # Integration Weights (Clenshaw-Curtis quadrature)
        # For N=5, simple weights are acceptable, but CC is better.
        # Approximation: Trapezoidal on the non-uniform grid
        self.weights = np.zeros(N_nodes + 1)
        for i in range(N_nodes):
            self.weights[i] += 0.5 * (self.nodes[i+1] - self.nodes[i])
            self.weights[i+1] += 0.5 * (self.nodes[i+1] - self.nodes[i])
            
    def _cheb_diff_matrix(self, N):
        """ Standard Chebyshev differentiation matrix on [0,1] """
        # 1. Compute on [-1, 1]
        x = np.cos(np.pi * np.arange(N + 1) / N)
        c = np.hstack([2, np.ones(N-1), 2]) * (-1)**np.arange(N+1)
        X = np.tile(x, (N+1, 1))
        dX = X.T - X
        D = (c[:, None] / c[None, :]) / (dX + np.eye(N+1))
        D = D - np.diag(np.sum(D.T, axis=0))
        # 2. Scale for [0, 1]: d/ds = d/dx * dx/ds = d/dx * 2
        return -2.0 * D

    def solve(self, x_start, x_end, warm_start=None, max_iter=2):
        """
        Solves Geodesic via Sequential Quadratic Programming (SQP).
        Min E = sum w_k * v_k^T W(x_k) v_k
        s.t. x_0 = start, x_N = end
        """
        # Initialize Path (x_nodes): Shape (n, N+1)
        if warm_start is not None:
            X = warm_start
            # Fix endpoints (warm start might have drifted)
            X[:, 0] = x_start
            X[:, -1] = x_end
        else:
            # Linear Interpolation
            X = np.zeros((self.n, self.N + 1))
            for i in range(self.N + 1):
                X[:, i] = x_start + self.nodes[i] * (x_end - x_start)

        # SQP Loop
        for _ in range(max_iter):
            # 1. Evaluate Metric W at current nodes
            # W_stack shape: (N+1, n, n)
            W_stack = np.zeros((self.N + 1, self.n, self.n))
            for k in range(self.N + 1):
                W_stack[k] = self.W_func(X[:, k])

            # 2. Formulate QP: Min 0.5 * x_flat.T * H * x_flat
            # H = sum_k w_k * (D_k^T \otimes I) * W_k * (D_k \otimes I)
            # This is simpler in matrix form:
            # E = sum w_k * v_k^T W_k v_k
            # Let V = X @ D^T.  Col k of V is v_k.
            # We want to solve for X_flat.
            
            # Construct Hessian Block Matrix (Iteratively or Sparse)
            # Size (n*(N+1)) x (n*(N+1))
            # H_big is block matrix.
            # Entry (node i, node j) is a (n x n) block.
            # Block_ij = sum_k (w_k * D_ki * D_kj * W_k)
            
            # Efficient Vectorized Construction:
            # H = (D^T @ W_diag @ D) \otimes I ?? No, W varies.
            # Correct: H = (D.T \otimes I) * BlockDiag(w_k W_k) * (D \otimes I)
            
            # Construct Block Diagonal W_weighted
            W_diag_list = [self.weights[k] * W_stack[k] for k in range(self.N + 1)]
            W_blk = scipy.linalg.block_diag(*W_diag_list) # Size n(N+1) x n(N+1)
            
            # Construct Big D matrix: v_flat = D_big @ x_flat
            # D_mat is (N+1)x(N+1). We need to Kronecker with I_n
            D_big = np.kron(self.D_mat, np.eye(self.n))
            
            # Full Hessian
            H = D_big.T @ W_blk @ D_big
            
            # 3. Solve Equality Constrained QP
            # H x + A.T lam = 0
            # A x = b
            # A fixes first and last nodes.
            # x_flat = [x0, x1, ... xN]
            
            # Constraints: x0 = x_start, xN = x_end
            # Indices:
            idx_start = np.arange(self.n)
            idx_end   = np.arange(self.n * self.N, self.n * (self.N + 1))
            
            # Eliminate constrained variables (Schur complement or partitioning) to speed up
            # Free indices: 1 to N-1
            idx_free = np.setdiff1d(np.arange(self.n * (self.N + 1)), 
                                    np.concatenate([idx_start, idx_end]))
            
            # Partition Hessian
            # H_ff (free-free), H_fc (free-constrained)
            H_ff = H[np.ix_(idx_free, idx_free)]
            H_fc = H[np.ix_(idx_free, idx_start)] # interactions with start
            H_fe = H[np.ix_(idx_free, idx_end)]   # interactions with end
            
            # RHS = - (H_fc @ x_start + H_fe @ x_end)
            rhs = - (H_fc @ x_start + H_fe @ x_end)
            
            # Solve Linear System for free nodes
            # Regularize H_ff slightly for stability
            H_ff += 1e-6 * np.eye(len(idx_free))
            
            x_free = scipy.linalg.solve(H_ff, rhs, assume_a='pos')
            
            # Reconstruct Full Path
            X_flat = np.zeros(self.n * (self.N + 1))
            X_flat[idx_start] = x_start
            X_flat[idx_end]   = x_end
            X_flat[idx_free]  = x_free
            
            # Update X for next iteration
            X = X_flat.reshape((self.N + 1, self.n)).T
            
        return X

    def get_path_data(self, c_opt):
        # c_opt is essentially X (state at nodes)
        X = c_opt
        # Derivatives v = X @ D.T
        V = X @ self.D_mat.T
        return X, V, self.weights
    

class GeodesicSolver:
    def __init__(self, n, N_nodes, W_func):
        """
        n: State dimension (12)
        N_nodes: Number of collocation nodes (e.g. 5 or 10)
        W_func: Function handle W(state_error) -> Matrix
        """
        self.n = n
        self.N = N_nodes
        self.W_func = W_func
        
        # Chebyshev Nodes (Clenshaw-Curtis)
        # Map [-1, 1] -> [0, 1]
        theta = np.pi * np.arange(0, N_nodes + 1) / N_nodes
        x_cheb = np.cos(theta)
        self.nodes = 0.5 * (1 - x_cheb) # s in [0, 1]
        
        # Clenshaw-Curtis Weights (simplified approximation or standard)
        # For N=5, simple weights are often sufficient, but let's use trapezoidal for robustness
        # or precomputed CC weights if precision is needed. 
        # Using Trapezoidal rule on the nodes for the integral
        self.weights = np.zeros(N_nodes + 1)
        # Note: Non-uniform grid requires careful weighting. 
        # Simple approach: derivative is computed analytically, integral is sum.
        
        # Chebyshev Differentiation Matrix (D)
        # Used to compute gamma_prime from gamma nodes
        self.D_mat = self._cheb_diff_matrix(N_nodes)
        
    def _cheb_diff_matrix(self, N):
        """Standard Chebyshev differentiation matrix"""
        x = np.cos(np.pi * np.arange(N + 1) / N)
        c = np.hstack([2, np.ones(N-1), 2]) * (-1)**np.arange(N+1)
        X = np.tile(x, (N+1, 1))
        dX = X.T - X
        D = (c[:, None] / c[None, :]) / (dX + np.eye(N+1))
        D = D - np.diag(np.sum(D.T, axis=0))
        
        # We need derivative w.r.t s in [0,1], not x in [-1,1]
        # s = (1-x)/2  => ds = -dx/2 => d/ds = -2 d/dx
        return -2.0 * D

    def solve(self, x_start, x_end, warm_start=None):
        """
        Finds gamma(s) minimizing Energy = int v^T W v ds
        """
        # Flat decision variables: (N+1) * n
        if warm_start is not None:
            c0 = warm_start
        else:
            # Linear interp initialization
            c_mat = np.zeros((self.n, self.N + 1))
            for i in range(self.N + 1):
                s = self.nodes[i]
                c_mat[:, i] = x_start + s * (x_end - x_start)
            c0 = c_mat.flatten()

        # Constraints: Start and End points fixed
        # indices for s=0 (node 0) and s=1 (node N)
        # Optimization vector structure: [x_node0, x_node1, ... x_nodeN]
        
        def constraints(c):
            C = c.reshape((self.n, self.N + 1))
            # Error at start and end
            eq1 = C[:, 0] - x_start
            eq2 = C[:, -1] - x_end
            return np.hstack([eq1, eq2])

        # Objective: Riemann Energy
        def energy_cost(c):
            C = c.reshape((self.n, self.N + 1))
            # Compute derivatives at all nodes: V = C @ D.T
            # Shape (n, N+1)
            V = C @ self.D_mat.T 
            
            total_E = 0
            # Integral approx (Trapezoidal on Chebyshev grid)
            for k in range(self.N):
                # Segment k to k+1
                ds = self.nodes[k+1] - self.nodes[k]
                
                # Average W and V^2
                # Note: This is an approximation. Strict CCM uses expensive quadrature.
                # Evaluation at midpoint or nodes.
                xk = C[:, k]
                vk = V[:, k]
                Wk = self.W_func(xk)
                ek = vk.T @ Wk @ vk
                
                xk_next = C[:, k+1]
                vk_next = V[:, k+1]
                Wk_next = self.W_func(xk_next)
                ek_next = vk_next.T @ Wk_next @ vk_next
                
                total_E += 0.5 * (ek + ek_next) * ds
            return total_E

        # Solve SQP
        # Eq constraints must be 0
        cons = {'type': 'eq', 'fun': constraints}
        
        # Fast optimization options
        res = minimize(energy_cost, c0, method='SLSQP', constraints=cons, 
                       options={'maxiter': 20, 'ftol': 1e-2, 'disp': False})
        
        return res.x

    def get_path_data(self, c_opt):
        C = c_opt.reshape((self.n, self.N + 1))
        V = C @ self.D_mat.T
        return C, V, self.nodes

# ==========================================
# 2. SIMULATOR (Unified + Geodesic)
# ==========================================
class RCCMSimulator:
    def __init__(self, W_coeffs, Y_coeffs, basis_exps):
        self.Ws = W_coeffs
        self.Ys = Y_coeffs
        self.exps = basis_exps
        
        # Auto-detect mode
        dim_basis = len(basis_exps[0])
        if dim_basis == 9: self.mode = 'ROTATION'
        elif dim_basis == 4: self.mode = 'QUATERNION'
        else: raise ValueError(f"Unknown basis: {dim_basis}")
        
        # Constants
        self.m = 1.0; self.g = 9.81
        self.J = np.diag([0.02, 0.02, 0.04])
        self.J_inv = np.linalg.inv(self.J)
        
        # Init Geodesic Solver
        # We need 5-10 nodes for decent accuracy without killing speed
        self.geo = FastGeodesicSolver(n=12, N_nodes=5, W_func=self._eval_W_for_solver)
        self.last_geo_sol = None
        
        # State Context for W_func (Updated every step)
        self.current_q_ref = np.array([1.,0.,0.,0.])

    def _eval_W_for_solver(self, x_error):
        """
        Maps the error state 'x_error' (12D) back to an absolute orientation
        to evaluate the Metric W(q) or W(R).
        
        x_error = [e_p, eta, e_v, e_w]
        We need 'q' or 'R' to evaluate polynomials.
        Assumption: q_actual ~= q_ref * exp(eta/2)
        """
        eta = x_error[3:6]
        
        # Map eta -> q_delta (Small angle approximation is fast & sufficient for W evaluation)
        # eta is approx rotation vector
        angle = np.linalg.norm(eta)
        if angle < 1e-6:
            q_delta = np.array([1., 0., 0., 0.])
        else:
            # eta is 2*axis*sin(theta/2) or similar depending on definition.
            # Using simple rotvec:
            q_delta = Rotation.from_rotvec(eta).as_quat() # [x,y,z,w]
            q_delta = q_delta[[3,0,1,2]] # [w,x,y,z]
            
        # q_eval = q_ref * q_delta
        # (Quaternion multiplication)
        w1,x1,y1,z1 = self.current_q_ref
        w2,x2,y2,z2 = q_delta
        q_eval = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        
        # Evaluate W at q_eval
        if self.mode == 'QUATERNION':
            return self._compute_W_core(q_eval)
        else:
            # Convert back to R for Rotation mode
            R_eval = Rotation.from_quat(q_eval[[1,2,3,0]]).as_matrix()
            return self._compute_W_core(R_eval.flatten())

    def _compute_W_core(self, state_vec):
        weights = []
        for exp in self.exps:
            val = 1.0
            for i, p in enumerate(exp):
                if p > 0: val *= (state_vec[i]**p)
            weights.append(val)
        W = sum(self.Ws[k] * w for k, w in enumerate(weights))
        return W + 1e-6*np.eye(12) # Regularize

    def _compute_matrices(self, state_vec):
        """Returns both W and Y"""
        weights = []
        for exp in self.exps:
            val = 1.0
            for i, p in enumerate(exp):
                if p > 0: val *= (state_vec[i]**p)
            weights.append(val)
        
        W = sum(self.Ws[k] * w for k, w in enumerate(weights)) + 1e-6*np.eye(12)
        Y = sum(self.Ys[k] * w for k, w in enumerate(weights))
        return W, Y

    def nominal_trajectory(self, t):
        # ... (Same Trajectory Logic as Previous - Yaw+Pitch) ...
        omega_traj = 1.0
        px, py = np.sin(omega_traj*t), np.sin(omega_traj*t/2); pz=1.0
        vx, vy = omega_traj*np.cos(omega_traj*t), (omega_traj/2)*np.cos(omega_traj*t/2); vz=0
        ax, ay = -(omega_traj**2)*np.sin(omega_traj*t), -(omega_traj/2)**2*np.sin(omega_traj*t/2); az=0
        
        f_nom = self.m*(np.array([ax,ay,az]) - np.array([0,0,-9.81]))
        
        psi, theta, phi = 0.5*t, 0.35*np.sin(2*t), 0.0
        R_ref = Rotation.from_euler('zyx', [psi, theta, phi]).as_matrix()
        
        # Angular Velocity (Body Frame)
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        
        # 1st Derivatives of Euler Angles
        dpsi = 0.5
        dtheta = 0.7 * np.cos(2*t)
        dphi = 0.0
        
        # 2nd Derivatives of Euler Angles (NEW)
        ddpsi = 0.0
        ddtheta = -1.4 * np.sin(2*t) # deriv of 0.7*cos(2t)
        ddphi = 0.0

        # w_ref (Body Rates)
        wx = dphi - dpsi * st
        wy = dtheta * cp + dpsi * ct * sp
        wz = dpsi * ct * cp - dtheta * sp
        w_ref = np.array([wx, wy, wz])
        
        # Angular Acceleration (dw_ref)
        # Computed by differentiating w_ref components w.r.t time
        # Chain rule: d(sin(q))/dt = cos(q)*dq
        
        # dwx = ddphi - (ddpsi*st + dpsi*ct*dtheta)
        dwx = ddphi - (ddpsi * st + dpsi * ct * dtheta)
        
        # dwy = (ddtheta*cp - dtheta*sp*dphi) + (ddpsi*ct*sp - dpsi*st*dtheta*sp + dpsi*ct*cp*dphi)
        term1_y = ddtheta * cp - dtheta * sp * dphi
        term2_y = ddpsi * ct * sp - dpsi * st * dtheta * sp + dpsi * ct * cp * dphi
        dwy = term1_y + term2_y
        
        # dwz = (ddpsi*ct*cp - dpsi*st*dtheta*cp - dpsi*ct*sp*dphi) - (ddtheta*sp + dtheta*cp*dphi)
        term1_z = ddpsi * ct * cp - dpsi * st * dtheta * cp - dpsi * ct * sp * dphi
        term2_z = ddtheta * sp + dtheta * cp * dphi
        dwz = term1_z - term2_z
        
        dw_ref = np.array([dwx, dwy, dwz])
        
        # Accurate Feedforward Torque
        # tau = J * dw + w x (J * w)
        tau_nom = self.J @ dw_ref + np.cross(w_ref, self.J @ w_ref)
        
        return np.hstack([px, py, pz]), R_ref, np.hstack([vx, vy, vz]), w_ref, np.hstack([f_nom, tau_nom])

    def run_simulation(self, duration=10.0, dt=0.001):
        steps = int(duration / dt)
        times = np.linspace(0, duration, steps)
        
        # Init State
        p0, R0, v0, w0, _ = self.nominal_trajectory(0)
        p_act = p0 + np.array([0.1, -0.1, 0.05])
        state = np.concatenate([p_act, R0.flatten(), v0, w0])
        
        # --- NEW LOGGING VARIABLE ---
        pos_hist, ref_hist, err_hist, att_hist, u_hist = [], [], [], [], []
        
        print(f">>> Starting Geodesic Simulation ({self.mode})...")
        
        for i, t in enumerate(tqdm(times)):
            # ... (Calculation of p, R, v, w, Error State, etc. remains same) ...
            p_r, R_r, v_r, w_r, u_nom = self.nominal_trajectory(t)
            p = state[0:3]; R = state[3:12].reshape((3,3)); v = state[12:15]; w = state[15:18]
            
            # Error Calcs
            e_p = p - p_r; e_v = v - v_r; e_w = w - w_r
            R_err_mat = R_r.T @ R
            rot_vec = Rotation.from_matrix(R_err_mat).as_rotvec()
            x_error_end = np.concatenate([e_p, rot_vec, e_v, e_w])
            
            # Context Update
            q_temp = Rotation.from_matrix(R_r).as_quat() 
            self.current_q_ref = q_temp[[3, 0, 1, 2]]
            if self.current_q_ref[0] < 0: self.current_q_ref *= -1
            
            # Geodesic Control
            c_opt = self.geo.solve(np.zeros(12), x_error_end, self.last_geo_sol)
            self.last_geo_sol = c_opt
            
            # Integral Feedback Loop (Same as before)
            gamma, gamma_s, weights_node = self.geo.get_path_data(c_opt)
            u_fb = np.zeros(6)
            for k in range(len(weights_node)):
                x_k = gamma[:, k]; v_k = gamma_s[:, k]; w_k = weights_node[k]
                W_k = self._eval_W_for_solver(x_k)
                
                eta_k = x_k[3:6]
                angle = np.linalg.norm(eta_k)
                if angle < 1e-6: q_d = np.array([1.,0.,0.,0.])
                else: q_d = Rotation.from_rotvec(eta_k).as_quat()[[3,0,1,2]]
                
                w1,x1,y1,z1 = self.current_q_ref; w2,x2,y2,z2 = q_d
                q_k = np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, 
                                w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
                
                if self.mode=='QUATERNION': _, Y_k = self._compute_matrices(q_k)
                else: _, Y_k = self._compute_matrices(Rotation.from_quat(q_k[[1,2,3,0]]).as_matrix().flatten())
                
                K_k = Y_k @ np.linalg.inv(W_k)
                u_fb += w_k * (K_k @ v_k)
            
            u_total = u_nom + np.clip(u_fb, -50, 50)
            
            # Disturbance
            dist = np.random.normal(0,0.2,6)
            if 2.0<t<4.0: dist[0]+=1.5; dist[3]+=0.1
            
            state_next = self.rk4_step(t, state, u_total, dist, dt)
            if np.linalg.norm(state_next[0:3]) > 100: break
            state = state_next
            
            U,_,Vt = np.linalg.svd(state[3:12].reshape((3,3)))
            state[3:12] = (U@Vt).flatten()
            
            # --- LOGGING UPDATES ---
            pos_hist.append(p)
            ref_hist.append(p_r)
            err_hist.append(np.linalg.norm(x_error_end))
            att_hist.append(Rotation.from_matrix(R_err_mat).as_euler('xyz', degrees=True))
            u_hist.append(u_total) # Log Inputs
            
        return np.array(pos_hist), np.array(ref_hist), times[:len(pos_hist)], \
               np.array(err_hist), np.array(att_hist), np.array(u_hist)

    # ... (Include dynamics/rk4_step from previous snippets) ...
    def dynamics(self, t, x_state, u, dist):
        R = x_state[3:12].reshape((3,3))
        v = x_state[12:15]
        w = x_state[15:18]
        
        f_W = np.clip(u[0:3] + dist[0:3], -50, 50)
        tau_B = np.clip(u[3:6] + dist[3:6], -10, 10)
        
        dp = v
        w_hat = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        dR = R @ w_hat
        dv = (1.0/self.m) * f_W + np.array([0, 0, -self.g])
        dw = self.J_inv @ (tau_B - np.cross(w, self.J @ w))
        return np.concatenate([dp, dR.flatten(), dv, dw])

    def rk4_step(self, t, state, u, dist, dt):
        k1 = self.dynamics(t, state, u, dist)
        k2 = self.dynamics(t+dt/2, state+dt/2*k1, u, dist)
        k3 = self.dynamics(t+dt/2, state+dt/2*k2, u, dist)
        k4 = self.dynamics(t+dt, state+dt*k3, u, dist)
        return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


if __name__ == "__main__":
    with open("rccm_controller_data.pkl", "rb") as f: data = pickle.load(f)
    sim = RCCMSimulator(data["W_matrices"], data["Y_matrices"], data["basis_exponents"])
    
    # Unpack 6 items now
    pos, ref, t, err, att, u_log = sim.run_simulation()
    
    # Pass u_log to plotter
    plot_rccm_results(pos, ref, t, err, att, u_log, tube_alpha=data.get("alpha", 0.5))
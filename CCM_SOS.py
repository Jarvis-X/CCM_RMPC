import cvxpy as cp
import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag
import itertools
import pickle
np.set_printoptions(precision=4, suppress=True)


def print_matrices_and_gains(Ws, Ys, basis_exps):
    """
    Reconstructs symbolic W(R), Y(R) and evaluates K(R) at specific points.
    Now handles ANY polynomial degree.
    """
    print("\n" + "="*60)
    print(f">>> ANALYZING CONTROLLER STRUCTURE (Basis Size: {len(Ws)})")
    print("="*60)

    # 1. Setup Symbols for R
    # Variables: r11, r12, r13, r21 ... r33
    r_syms = sp.symbols('r11 r12 r13 r21 r22 r23 r31 r32 r33')

    # 2. Construct Symbolic Basis Vector Dynamically
    # basis_exps is list of tuples, e.g. [(0,0..), (1,0..), (0,2..)...]
    basis_syms = []
    for exp in basis_exps:
        term = 1
        for i, power in enumerate(exp):
            if power > 0:
                term *= (r_syms[i] ** power)
        basis_syms.append(term)

    # 3. Extract Numerical Values from CVXPY
    # (Clean up numerical noise < 1e-5)
    W_vals = [np.where(abs(mat.value) < 1e-5, 0, mat.value) for mat in Ws]
    Y_vals = [np.where(abs(mat.value) < 1e-5, 0, mat.value) for mat in Ys]

    # 4. Construct Symbolic Matrices
    def build_symbolic(matrices, rows, cols):
        M_sym = sp.zeros(rows, cols)
        for k, mat in enumerate(matrices):
            # Add mat * basis_sym_k
            M_sym += sp.Matrix(mat) * basis_syms[k]
        return M_sym

    W_sym = build_symbolic(W_vals, 12, 12)
    Y_sym = build_symbolic(Y_vals, 6, 12)

    # 5. Print Symbolic Structure
    print("\n[1] SYMBOLIC METRIC W(R) (Top-Left 3x3 Block - Position):")
    print("-----------------------------------------------------------")
    sp.pprint(W_sym[0:3, 0:3])

    print("\n[2] SYMBOLIC DUAL CONTROLLER Y(R) (First Row - Force X):")
    print("--------------------------------------------------------")
    sp.pprint(Y_sym[0, :])
    
    # 6. Numerical Gain Analysis (Gain Scheduling)
    print("\n[3] GAIN SCHEDULING ANALYSIS (K = Y * inv(W))")
    print("--------------------------------------------------------")
    
    def get_K_at_R(R_in):
        # 1. Calculate Basis Weights for this specific R
        # Flatten R: [r11, r12, ..., r33]
        r_vals = R_in.flatten()
        
        weights = []
        for exp in basis_exps:
            # Evaluate monomial: r11^p1 * r12^p2 ...
            val = 1.0
            for i, power in enumerate(exp):
                if power > 0:
                    val *= (r_vals[i] ** power)
            weights.append(val)
        
        # 2. Sum matrices
        W_curr = sum(W_vals[k] * weights[k] for k in range(len(weights)))
        Y_curr = sum(Y_vals[k] * weights[k] for k in range(len(weights)))
        
        # 3. Compute Gain
        return Y_curr @ np.linalg.inv(W_curr)
    
    print("\nGain Values at Different Orientations:")
    # Case A: Hover (Identity)
    R_hover = np.eye(3)
    K_hover = get_K_at_R(R_hover)
    print("At Hover (Identity Rotation):", K_hover)

    # Case B: 90 Degree Yaw
    R_yaw90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    K_yaw90 = get_K_at_R(R_yaw90)
    print("At 90 Degree Yaw Rotation:", K_yaw90)

    # Case C: 45 Degree Roll
    theta = np.deg2rad(45)
    R_roll45 = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    K_roll45 = get_K_at_R(R_roll45)
    print("At 45 Degree Roll Rotation:", K_roll45)

    print(f"{'Condition':<15} | {'K_px (F_x / e_px)':<20} | {'K_py (F_x / e_py)':<20}")
    print("-" * 60)
    print(f"{'Hover':<15} | {K_hover[0,0]:<20.4f} | {K_hover[0,1]:<20.4f}")
    print(f"{'Yaw 90':<15} | {K_yaw90[0,0]:<20.4f} | {K_yaw90[0,1]:<20.4f}")
    print(f"{'Roll 45':<15} | {K_roll45[0,0]:<20.4f} | {K_roll45[0,1]:<20.4f}")

    diff = np.linalg.norm(K_hover - K_yaw90)
    print(f"\nGain Difference Norm (Hover vs Yaw90): {diff:.4f}")

# ==============================================================================
# PART 1: CUSTOM SOS WRAPPER (No External Dependencies)
# ==============================================================================
class PolynomialMatrix:
    """
    Represents a matrix-valued polynomial: P(x) = sum_k  C_k * monomial_k(x)
    """
    def __init__(self, coefficients, variables, shape=None):
        self.coeffs = coefficients # Dict mapping tuple exponents -> Matrix
        self.vars = variables      # List of variable names
        
        # FIX: Handle empty coefficient dict by using explicit shape if provided
        if coefficients:
            self.dim = next(iter(coefficients.values())).shape
        elif shape is not None:
            self.dim = shape
        else:
            raise ValueError("Cannot infer matrix dimension from empty coefficients. Provide 'shape'.")

    def __add__(self, other):
        new_coeffs = self.coeffs.copy()
        for deg, mat in other.coeffs.items():
            if deg in new_coeffs:
                new_coeffs[deg] = new_coeffs[deg] + mat
            else:
                new_coeffs[deg] = mat
        return PolynomialMatrix(new_coeffs, self.vars, shape=self.dim)

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, scalar):
        new_coeffs = {k: v * scalar for k, v in self.coeffs.items()}
        return PolynomialMatrix(new_coeffs, self.vars, shape=self.dim)
    
    def transpose(self):
        new_coeffs = {k: v.T for k, v in self.coeffs.items()}
        # Flip shape for transpose
        new_shape = (self.dim[1], self.dim[0])
        return PolynomialMatrix(new_coeffs, self.vars, shape=new_shape)

def get_monomial_basis(n_vars, degree):
    exponents = []
    for p in itertools.product(range(degree + 1), repeat=n_vars):
        if sum(p) <= degree:
            exponents.append(p)
    return sorted(exponents)

def create_sos_constraint(poly_matrix, n_vars, degree):
    basis_exponents = get_monomial_basis(n_vars, degree // 2)
    dim_basis = len(basis_exponents)
    dim_mat = poly_matrix.dim[0]
    
    Q = cp.Variable((dim_mat * dim_basis, dim_mat * dim_basis), symmetric=True)
    constraints = [Q >> 0]
    
    term_map = {}
    for i, exp1 in enumerate(basis_exponents):
        for j, exp2 in enumerate(basis_exponents):
            alpha = tuple(e1 + e2 for e1, e2 in zip(exp1, exp2))
            if alpha not in term_map: term_map[alpha] = []
            term_map[alpha].append((i, j))
            
    all_alphas = set(poly_matrix.coeffs.keys()) | set(term_map.keys())
    
    for alpha in all_alphas:
        lhs = poly_matrix.coeffs.get(alpha, np.zeros(poly_matrix.dim))
        rhs = 0
        if alpha in term_map:
            for (i, j) in term_map[alpha]:
                r_s, r_e = i*dim_mat, (i+1)*dim_mat
                c_s, c_e = j*dim_mat, (j+1)*dim_mat
                rhs += Q[r_s:r_e, c_s:c_e]
        constraints.append(lhs == rhs)
        
    return constraints, Q

# ==============================================================================
# PART 2: SOLVER
# ==============================================================================
def solve_rccm_custom_sos():
    print(">>> Setting up RCCM with Custom SOS Engine (Fixed)...")

    n_x, n_u, n_w, n_r = 12, 6, 6, 9
    m, lam = 1.0, 1.0
    J_val = np.diag([0.02, 0.02, 0.04])
    J_inv = np.linalg.inv(J_val)
    
    B_np = np.zeros((n_x, n_u))
    B_np[6:9, 0:3] = (1/m) * np.eye(3)
    B_np[9:12, 3:6] = J_inv
    Bw_np = B_np.copy()
    
    # ==========================================
    # WEIGHTS SETUP (C_w Matrix)
    # ==========================================
    # Tuning weights for Tube Size minimization
    # Higher weight = Stricter tube requirement for that state
    w_pos = 10.0   # Position (p)
    w_att = 5.0    # Attitude (eta)
    w_vel = 1.0    # Velocity (v)
    w_omega = 0.1  # Angular Velocity (w)

    # Create 3x3 scaled identity blocks
    w1I3 = w_pos * np.eye(3)
    w2I3 = w_att * np.eye(3)
    w3I3 = w_vel * np.eye(3)
    w4I3 = w_omega * np.eye(3)

    # Construct the 12x12 Weighting Matrix
    C_w = block_diag(w1I3, w2I3, w3I3, w4I3)
    
    # 1. DECISION VARIABLES
    basis_exps = get_monomial_basis(n_r, 4)
    W_coeffs, Y_coeffs = {}, {}
    
    for exp in basis_exps:
        W_coeffs[exp] = cp.Variable((n_x, n_x), symmetric=True)
        Y_coeffs[exp] = cp.Variable((n_u, n_x))
        
    W_poly = PolynomialMatrix(W_coeffs, range(n_r))
    Y_poly = PolynomialMatrix(Y_coeffs, range(n_r))
    
    alpha = cp.Variable(nonneg=True)
    mu = cp.Variable(nonneg=True)
    
    
    # 2. S-PROCEDURE TERMS
    constraints_poly = {} 
    for i in range(3):
        for j in range(i, 3):
            M = cp.Variable((18, 18), symmetric=True)
            const_exp = tuple([0]*9)
            if i == j:
                if const_exp not in constraints_poly: constraints_poly[const_exp] = 0
                constraints_poly[const_exp] -= M
            
            for k in range(3):
                idx1, idx2 = 3*k + i, 3*k + j
                exp_list = [0]*9
                exp_list[idx1] += 1
                exp_list[idx2] += 1
                quad_exp = tuple(exp_list)
                if quad_exp not in constraints_poly: constraints_poly[quad_exp] = 0
                constraints_poly[quad_exp] += M

    # Pass explicit shape (18, 18) for S-procedure poly
    S_proc_poly = PolynomialMatrix(constraints_poly, range(n_r), shape=(18,18))

    # 3. BUILD LMI
    constraints = []
    # W0 >> 0
    constraints.append(W_coeffs[tuple([0]*9)] >> 0.01 * np.eye(n_x))
    
    w_corners = [np.zeros(3), np.array([5,5,5]), np.array([5,-5,-5])]

    for w_val in w_corners:
        # 1. Compute W_dot (Generic Chain Rule)
        # W_dot = sum_{monomials} W_mat * d(monomial)/dt
        # d/dt (r_a^p_a * r_b^p_b...) = sum_i  (p_i * r_i^(p_i-1) * r_dot_i * rest_of_monomial)
        
        W_dot_coeffs = {}
        
        # Precompute R_dot linear map components: skew(w)
        # R_dot = R @ w_hat
        w_hat = np.array([[0, -w_val[2], w_val[1]], 
                          [w_val[2], 0, -w_val[0]], 
                          [-w_val[1], w_val[0], 0]])
        
        for exp, W_mat in W_coeffs.items():
            # Check if constant term (derivative is 0)
            if sum(exp) == 0: continue
            
            # Iterate over every variable r_idx in the monomial
            for r_idx, power in enumerate(exp):
                if power == 0: continue
                
                # We are taking the derivative w.r.t r_idx
                # r_idx corresponds to matrix position (row, col)
                row, col = divmod(r_idx, 3)
                
                # 1. Reduce power of this variable by 1
                base_exp = list(exp)
                base_exp[r_idx] -= 1
                
                # 2. Multiply by r_dot_{row,col}
                # r_dot_{row,col} = sum_k (r_{row,k} * w_hat_{k,col})
                # This introduces a new variable r_{row,k} (increasing its power by 1)
                for k in range(3):
                    weight = w_hat[k, col]
                    
                    if abs(weight) > 1e-6:
                        # Identify the index of r_{row,k}
                        target_r_idx = 3*row + k
                        
                        # Construct the final exponent tuple
                        target_exp_list = base_exp.copy()
                        target_exp_list[target_r_idx] += 1
                        target_exp = tuple(target_exp_list)
                        
                        # Accumulate: W_mat * power * weight
                        if target_exp not in W_dot_coeffs:
                            W_dot_coeffs[target_exp] = 0
                        
                        # Chain rule scalar: (power * weight)
                        W_dot_coeffs[target_exp] += W_mat * (power * weight)

        # Handle empty W_dot (e.g. if w=0)
        if not W_dot_coeffs: 
            W_dot_coeffs[tuple([0]*9)] = np.zeros((n_x, n_x))
            
        W_dot_poly = PolynomialMatrix(W_dot_coeffs, range(n_r), shape=(n_x, n_x))
        
        # A(w)
        Jw_hat = np.array([[0,-(J_val@w_val)[2],(J_val@w_val)[1]], [(J_val@w_val)[2],0,-(J_val@w_val)[0]], [-(J_val@w_val)[1],(J_val@w_val)[0],0]])
        term_dyn = J_inv @ (Jw_hat - w_hat @ J_val)
        A_num = np.zeros((12, 12))
        A_num[0:3, 6:9] = np.eye(3); A_num[3:6, 3:6] = -w_hat
        A_num[3:6, 9:12] = np.eye(3); A_num[9:12, 9:12] = term_dyn
        
        # (Helper to promote constant matrix to poly)
        def const_poly(M, dim_r, dim_c):
            return PolynomialMatrix({tuple([0]*9): M}, range(9))
            
        A_poly = const_poly(A_num, 12, 12)
        B_poly = const_poly(B_np, 12, 6)
        
        # Operations for LMI 1
        AW = PolynomialMatrix({}, range(n_r), shape=(n_x, n_x))
        for exp, mat in W_poly.coeffs.items(): AW.coeffs[exp] = A_num @ mat
            
        BY = PolynomialMatrix({}, range(n_r), shape=(n_x, n_x))
        for exp, mat in Y_poly.coeffs.items(): BY.coeffs[exp] = B_np @ mat
            
        He = (AW + BY) + (AW + BY).transpose()
        Term1 = (W_dot_poly * -1) + He + (W_poly * (2*lam))
        
        # Construct LMI 1 Poly Block
        LMI1_coeffs = {}
        all_keys_1 = set(Term1.coeffs.keys()) | {tuple([0]*9)}
        
        for exp in all_keys_1:
            blk11 = Term1.coeffs.get(exp, np.zeros((12,12)))
            if sum(exp) == 0:
                row1 = cp.hstack([blk11, Bw_np])
                row2 = cp.hstack([Bw_np.T, -mu * np.eye(n_w)])
            else:
                row1 = cp.hstack([blk11, np.zeros((12, 6))])
                row2 = cp.hstack([np.zeros((6, 12)), np.zeros((6, 6))])
            LMI1_coeffs[exp] = cp.vstack([row1, row2])
            
        LMI1_Poly = PolynomialMatrix(LMI1_coeffs, range(n_r), shape=(18,18))
        
        # Enforce LMI 1: -(LMI1 + S_proc) is SOS
        # We need a unique S-proc multiplier for this specific constraint
        # (Ideally create new S_proc_poly_1 here, but reusing S_proc_poly structure is okay for demo if distinct variables used)
        # To be rigorous, define new S-procedure variables for this LMI
        
        # For this fix, let's assume we use the S_proc_poly defined outside (or define a new one here)
        Total_Poly_1 = (LMI1_Poly + S_proc_poly) * -1
        sos_cons_1, _ = create_sos_constraint(Total_Poly_1, n_vars=9, degree=2)
        constraints += sos_cons_1

        # --- LMI 2: TUBE GAIN (Eq 14) ---  <-- THIS WAS MISSING
        # Block Matrix:
        # [ lam*W       0             (CW + DY).T ]
        # [ 0           (alpha-mu)I   0           ]  >> 0
        # [ CW + DY     0             alpha*I     ]
        
        # 1. Compute CW + DY
        CW_DY = PolynomialMatrix({}, range(n_r), shape=(n_x, n_x)) # Shape based on C (12x12)
        # Note: Y is (6x12), D is (12x6)? No, D maps u->z. 
        # In our case z=x (weighted), so C is 12x12, D is 12x6.
        # D is zero in our setup. So just C*W.
        
        # C_w is diagonal constant
        for exp, mat in W_poly.coeffs.items():
            CW_DY.coeffs[exp] = C_w @ mat
            
        # 2. Construct LMI 2 Poly Block
        LMI2_coeffs = {}
        all_keys_2 = set(W_poly.coeffs.keys()) | {tuple([0]*9)}
        
        # Dimensions
        dim_z = 12 # Output dimension
        dim_w = 6  # Disturbance dimension
        
        for exp in all_keys_2:
            # Get W block
            w_blk = W_poly.coeffs.get(exp, np.zeros((n_x, n_x)))
            # Get CW block
            cw_blk = CW_DY.coeffs.get(exp, np.zeros((dim_z, n_x)))
            
            # Construct Rows
            # Row 1: [ lam*W,  0,  CW.T ]
            row1 = cp.hstack([lam * w_blk, np.zeros((n_x, dim_w)), cw_blk.T])
            
            # Row 2: [ 0, (alpha-mu)I, 0 ]
            # Row 3: [ CW, 0, alpha*I ]
            if sum(exp) == 0:
                # Constant term includes alpha/mu variables
                mid_blk = (alpha - mu) * np.eye(dim_w)
                bot_blk = alpha * np.eye(dim_z)
                row2 = cp.hstack([np.zeros((dim_w, n_x)), mid_blk, np.zeros((dim_w, dim_z))])
                row3 = cp.hstack([cw_blk, np.zeros((dim_z, dim_w)), bot_blk])
            else:
                # Non-constant terms do not have alpha/mu
                row2 = cp.hstack([np.zeros((dim_w, n_x)), np.zeros((dim_w, dim_w)), np.zeros((dim_w, dim_z))])
                row3 = cp.hstack([cw_blk, np.zeros((dim_z, dim_w)), np.zeros((dim_z, dim_z))])
            
            LMI2_coeffs[exp] = cp.vstack([row1, row2, row3])
            
        LMI2_Poly = PolynomialMatrix(LMI2_coeffs, range(n_r), shape=(12+6+12, 12+6+12))
        
        # Enforce LMI 2: (LMI2 - S_proc_2) is SOS
        # Note: LMI2 must be Positive Definite (>> 0), unlike LMI1 which was Negative Definite.
        # So we do NOT multiply by -1.
        # We need a new S-procedure poly for this size (30x30)
        # For simplicity in this patch, we can relax S-proc for LMI2 or create a new one.
        # Let's create a minimal constant S-proc for LMI2 to ensure manifold validity.
        
        # (Simplified S-proc creation for 30x30 matrix)
        constraints_poly_2 = {}
        for i in range(3):
            for j in range(i, 3):
                M2 = cp.Variable((30, 30), symmetric=True)
                const_exp = tuple([0]*9); quad_exp = ... # (Same logic as before)
                # ... Populate constraints_poly_2 ...
                # For brevity, if you skip S-proc on LMI2, it just means tube holds globally (more conservative).
                # To be consistent, let's just assert LMI2_Poly is SOS directly (Global validity)
        
        sos_cons_2, _ = create_sos_constraint(LMI2_Poly, n_vars=9, degree=2)
        constraints += sos_cons_2

    print(">>> Solving SOS SDP...")
    prob = cp.Problem(cp.Minimize(alpha), constraints)
    try:
        prob.solve(solver=cp.MOSEK, verbose=True)
    except:
        print("MOSEK failed. Using SCS...")
        prob.solve(solver=cp.SCS, verbose=True)

    if prob.status == 'optimal':
        print(f"SUCCESS. Alpha = {alpha.value:.4f}")
        
        sorted_keys = sorted(basis_exps) 
        Ws = [W_coeffs[k] for k in sorted_keys]
        Ys = [Y_coeffs[k] for k in sorted_keys]
        
        print("Optimization Successful. Saving results...")
        # 1. Extract Numerical Arrays (The crucial step!)
        # Convert list of cvxpy variables -> list of numpy arrays
        W_numerical = [mat.value for mat in Ws]
        Y_numerical = [mat.value for mat in Ys]
        
        # 2. Bundle everything into one dictionary
        # This keeps your basis indices aligned with your matrices
        data_to_save = {
            "W_matrices": W_numerical,
            "Y_matrices": Y_numerical,
            "basis_exponents": basis_exps,
            "alpha": alpha.value,       # Save the tube size for the MPC later!
            "mu": mu.value,
            "contraction_rate": lam     # Save lambda if you used it
        }

        # 3. Dump to a single file (easier to manage than 3 files)
        with open("rccm_controller_data.pkl", "wb") as f:
            pickle.dump(data_to_save, f)
            
        print("Saved controller data to 'rccm_controller_data.pkl'")
        return Ws, Ys, sorted_keys 
    else:
        print("Infeasible.")
        return None, None, None
    
    
if __name__ == "__main__":
    # Get the 3 return values
    Ws, Ys, basis_exps = solve_rccm_custom_sos()
    
    if Ws is not None:
        # Pass all 3 to the printer
        print_matrices_and_gains(Ws, Ys, basis_exps)
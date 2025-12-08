import cvxpy as cp
import numpy as np
import sympy as sp
import itertools
import pickle
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

# ==============================================================================
# SHARED UTILITIES
# ==============================================================================

def get_monomial_basis(n_vars, degree):
    """Generates sorted exponent tuples for n_vars up to degree."""
    exponents = []
    for p in itertools.product(range(degree + 1), repeat=n_vars):
        if sum(p) <= degree:
            exponents.append(p)
    return sorted(exponents)
# ==============================================================================
# PART 1: CUSTOM SOS ENGINE (Polynomial Arithmetic)
# ==============================================================================
class PolynomialMatrix:
    """
    Sparse Matrix-Valued Polynomial: P(x) = sum C_k * monomial_k(x)
    """
    def __init__(self, coefficients, n_vars, shape=None):
        self.coeffs = coefficients # Dict[tuple_deg, Matrix]
        self.n_vars = n_vars
        
        # Determine shape from first non-empty coefficient
        if coefficients:
            self.dim = next(iter(coefficients.values())).shape
        elif shape is not None:
            self.dim = shape
        else:
            raise ValueError("Need explicit shape for empty polynomial.")

    def __add__(self, other):
        new_coeffs = self.coeffs.copy()
        for deg, mat in other.coeffs.items():
            if deg in new_coeffs: new_coeffs[deg] = new_coeffs[deg] + mat
            else: new_coeffs[deg] = mat
        return PolynomialMatrix(new_coeffs, self.n_vars, shape=self.dim)

    def __sub__(self, other): return self + (other * -1)
    
    def __mul__(self, other):
        # 1. Scalar * PolyMatrix
        if isinstance(other, (int, float, cp.Variable, cp.Expression, np.ndarray)):
            # If multiplying by a matrix (constant), check dimensions
            if isinstance(other, np.ndarray) and other.shape == self.dim:
                # Elementwise or Matrix product? Assume Matrix Product if shapes align
                # But here we usually do Scalar * Matrix. 
                # Let's support Matrix @ Poly (Left Multiply) via a helper if needed.
                # For Scalar:
                pass
            new_coeffs = {k: v * other for k, v in self.coeffs.items()}
            return PolynomialMatrix(new_coeffs, self.n_vars, shape=self.dim)
            
        # 2. PolyScalar * PolyMatrix (Convolution)
        elif isinstance(other, PolynomialMatrix):
            if other.dim != (1,1): raise ValueError("Only Scalar * Matrix poly supported")
            new_coeffs = {}
            for exp1, mat1 in self.coeffs.items():
                for exp2, mat2 in other.coeffs.items():
                    new_exp = tuple(e1 + e2 for e1, e2 in zip(exp1, exp2))
                    term = mat1 * mat2
                    if new_exp not in new_coeffs: new_coeffs[new_exp] = term
                    else: new_coeffs[new_exp] += term
            return PolynomialMatrix(new_coeffs, self.n_vars, shape=self.dim)

    def transpose(self):
        new_coeffs = {k: v.T for k, v in self.coeffs.items()}
        return PolynomialMatrix(new_coeffs, self.n_vars, shape=(self.dim[1], self.dim[0]))
    
    @staticmethod
    def from_dict(coeffs, n_vars, shape):
        return PolynomialMatrix(coeffs, n_vars, shape)

def get_basis(n_vars, degree):
    """Generate exponents for all monomials up to 'degree'."""
    return sorted([p for p in itertools.product(range(degree + 1), repeat=n_vars) if sum(p) <= degree])

def sos_constraint(poly, degree):
    """
    Creates LMI constraints: poly(x) = z(x)^T Q z(x)
    """
    basis = get_basis(poly.n_vars, degree // 2)
    n_basis = len(basis)
    dim = poly.dim[0]
    
    # Gram Matrix
    Q = cp.Variable((dim * n_basis, dim * n_basis), symmetric=True)
    cons = [Q >> 0]
    
    # Map monomials to Gram blocks
    gram_map = {}
    for i, e1 in enumerate(basis):
        for j, e2 in enumerate(basis):
            deg = tuple(a+b for a,b in zip(e1, e2))
            if deg not in gram_map: gram_map[deg] = []
            gram_map[deg].append((i,j))
            
    # Coefficient Matching
    all_degs = set(poly.coeffs.keys()) | set(gram_map.keys())
    for deg in all_degs:
        lhs = poly.coeffs.get(deg, np.zeros(poly.dim))
        rhs = 0
        if deg in gram_map:
            for (i, j) in gram_map[deg]:
                # Extract block (i,j) from Q
                rhs += Q[i*dim:(i+1)*dim, j*dim:(j+1)*dim]
        cons.append(lhs == rhs)
        
    return cons

# ==============================================================================
# PART 2: UNIFIED SOLVER (ROTATION & QUATERNION)
# ==============================================================================

def solve_rccm_sos(mode='ROTATION', degree=2):
    print(f"\n>>> SETUP: {mode} RCCM (Degree {degree}, Full SOS)...")
    
    # 1. MODE CONFIGURATION
    if mode == 'ROTATION':
        n_r = 9  # r11..r33
        basis_exps = get_basis(n_r, degree) # W(R)
        mult_basis = get_basis(n_r, max(0, degree - 2)) # M(R)
    else:
        n_r = 4  # qw, qx, qy, qz
        basis_exps = get_basis(n_r, degree)
        mult_basis = get_basis(n_r, max(0, degree - 2))

    print(f"Basis Size: {len(basis_exps)} terms. Gram Matrix Size: {len(get_basis(n_r, degree//2))}^2 blocks.")

    # 2. SYSTEM CONSTANTS
    n_x, n_u, n_w = 12, 6, 6
    m, lam = 1.0, 1.0
    J_inv = np.linalg.inv(np.diag([0.02, 0.02, 0.04]))
    
    B_np = np.zeros((n_x, n_u)); B_np[6:9,0:3]=1/m*np.eye(3); B_np[9:12,3:6]=J_inv
    Bw_np = B_np.copy()
    
    C_w = block_diag(10*np.eye(3), 5*np.eye(3), 1*np.eye(3), 0.1*np.eye(3))

    # 3. DECISION VARIABLES
    W_coeffs = {e: cp.Variable((n_x,n_x), symmetric=True) for e in basis_exps}
    Y_coeffs = {e: cp.Variable((n_u,n_x)) for e in basis_exps}
    alpha = cp.Variable(nonneg=True)
    mu = cp.Variable(nonneg=True)
    
    W_poly = PolynomialMatrix(W_coeffs, n_r, (n_x,n_x))
    Y_poly = PolynomialMatrix(Y_coeffs, n_r, (n_u,n_x))

    # 4. S-PROCEDURE TERMS
    # We construct "g(x) = 0" constraints
    g_polys = []
    
    if mode == 'ROTATION':
        # 6 Constraints: (R^T R - I)_ij = 0 for i <= j
        # Map flat index 0..8 to (row, col)
        # r_idx = 3*i + j
        for i in range(3):
            for j in range(i, 3):
                # Poly = sum_k (r_ki * r_kj) - delta_ij
                c = {}
                c[tuple([0]*9)] = np.array([[-1.0 if i==j else 0.0]])
                for k in range(3):
                    e = [0]*9
                    e[3*k + i] += 1
                    e[3*k + j] += 1
                    e = tuple(e)
                    c[e] = c.get(e, 0) + 1.0
                g_polys.append(PolynomialMatrix(c, n_r, (1,1)))
                
    else: # QUATERNION
        # 1 Constraint: q^T q - 1 = 0
        c = {tuple([0]*4): np.array([[-1.0]])}
        for k in range(4):
            e = [0]*4; e[k]=2; c[tuple(e)] = np.array([[1.0]])
        g_polys.append(PolynomialMatrix(c, n_r, (1,1)))

    # Create Multipliers M_i(x) for each constraint
    # We need separate multipliers for LMI1 (Stability) and LMI2 (Tube)
    S1_term = PolynomialMatrix({}, n_r, (18,18))
    S2_term = PolynomialMatrix({}, n_r, (30,30))
    
    for g in g_polys:
        # Multiplier variables
        M1_c = {e: cp.Variable((18,18), symmetric=True) for e in mult_basis}
        M2_c = {e: cp.Variable((30,30), symmetric=True) for e in mult_basis}
        
        M1 = PolynomialMatrix(M1_c, n_r, (18,18))
        M2 = PolynomialMatrix(M2_c, n_r, (30,30))
        
        S1_term = S1_term + (M1 * g)
        S2_term = S2_term + (M2 * g)

    # 5. BUILD LMIs (Iterate Omega Vertices)
    constraints = []
    # W0 >> 0
    constraints.append(W_coeffs[tuple([0]*n_r)] >> 0.01 * np.eye(n_x))
    
    w_corners = [np.zeros(3), np.array([5,5,5]), np.array([5,-5,-5])]
    print("Building Constraints...")

    for w_val in w_corners:
        # --- A. W_dot Calculation ---
        # 1. Define Linear Map: x_dot = L @ x
        if mode == 'ROTATION':
            # r_dot = r @ w_hat.  (Using flat indices)
            # w_hat is 3x3. We need 9x9 map.
            w_hat = np.array([[0,-w_val[2],w_val[1]],[w_val[2],0,-w_val[0]],[-w_val[1],w_val[0],0]])
            # R_dot = R * w_hat.  Row i of R_dot depends on Row i of R.
            # R_dot_ij = sum_k R_ik * w_hat_kj
            # This is block diagonal in the flattened vector.
            L_map = np.kron(np.eye(3), w_hat.T) # Check Kronecker order? 
            # R_vec_dot = (I kron w_hat.T) R_vec
            # Let's stick to explicit indexing to be safe.
        else:
            wx,wy,wz=w_val
            L_map = 0.5 * np.array([[0,-wx,-wy,-wz],[wx,0,wz,-wy],[wy,-wz,0,wx],[wz,wy,-wx,0]])

        # 2. Compute W_dot
        W_dot_c = {}
        for exp, W_mat in W_coeffs.items():
            if sum(exp)==0: continue
            for i, p in enumerate(exp): # Variable x_i
                if p > 0:
                    base = list(exp); base[i] -= 1
                    # Chain rule: sum_j L[i,j] * x_j
                    for j in range(n_r):
                        weight = L_map[i, j] if mode=='QUATERNION' else \
                                 (w_hat[j%3, i%3] if i//3 == j//3 else 0) 
                                 # For Rotation: R_dot = R w_hat. element (r,c) -> sum_k r(r,k) w(k,c)
                                 # var i is (r,c). Depend on (r,k).
                        
                        if mode == 'ROTATION':
                            r, c = divmod(i, 3)
                            # R_dot_rc = sum_k R_rk * w_kc
                            for k in range(3):
                                weight = w_hat[k, c]
                                if abs(weight) > 1e-6:
                                    tgt_idx = 3*r + k
                                    tgt = base.copy(); tgt[tgt_idx] += 1; tgt = tuple(tgt)
                                    if tgt not in W_dot_c: W_dot_c[tgt] = 0
                                    W_dot_c[tgt] += W_mat * (p * weight)
                        else:
                            if abs(weight) > 1e-6:
                                tgt = base.copy(); tgt[j] += 1; tgt = tuple(tgt)
                                if tgt not in W_dot_c: W_dot_c[tgt] = 0
                                W_dot_c[tgt] += W_mat * (p * weight)

        # Fill missing zeros
        if not W_dot_c: W_dot_c[tuple([0]*n_r)] = np.zeros((n_x,n_x))
        for k in W_dot_c:
            if isinstance(W_dot_c[k], (int,float)): W_dot_c[k] = np.zeros((n_x,n_x))
            
        W_dot = PolynomialMatrix(W_dot_c, n_r, (n_x,n_x))

        # --- B. System Dynamics A(x) ---
        def skew(v): return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        w_h = skew(w_val); Jw_h = skew(np.diag([0.02,0.02,0.04])@w_val)
        term44 = np.linalg.inv(np.diag([0.02,0.02,0.04])) @ (Jw_h - w_h @ np.diag([0.02,0.02,0.04]))
        A0 = np.zeros((12,12)); A0[0:3,6:9]=np.eye(3); A0[3:6,3:6]=-w_h; A0[3:6,9:12]=np.eye(3); A0[9:12,9:12]=term44
        
        # --- C. LMI 1 (Stability) ---
        # Helper manual multiplication
        AW_c = {}; BY_c = {}
        for e, m in W_poly.coeffs.items(): AW_c[e] = A0 @ m
        for e, m in Y_poly.coeffs.items(): BY_c[e] = B_np @ m
        AW = PolynomialMatrix(AW_c, n_r, (12,12))
        BY = PolynomialMatrix(BY_c, n_r, (12,12))
        
        He = (AW + BY) + (AW + BY).transpose()
        Term1 = (W_dot * -1) + He + (W_poly * (2*lam))
        
        # Embed
        LMI1_c = {}
        keys = set(Term1.coeffs.keys()) | {tuple([0]*n_r)}
        for e in keys:
            blk = Term1.coeffs.get(e, np.zeros((12,12)))
            if sum(e)==0:
                r1 = cp.hstack([blk, Bw_np])
                r2 = cp.hstack([Bw_np.T, -mu*np.eye(6)])
            else:
                r1 = cp.hstack([blk, np.zeros((12,6))])
                r2 = cp.hstack([np.zeros((6,12)), np.zeros((6,6))])
            LMI1_c[e] = cp.vstack([r1,r2])
        LMI1 = PolynomialMatrix(LMI1_c, n_r, (18,18))
        
        # Enforce SOS: -(LMI1 + S1) >> 0
        constraints += sos_constraint((LMI1 + S1_term) * -1, degree)

        # --- D. LMI 2 (Tube) ---
        CW_c = {}
        for e, m in W_poly.coeffs.items(): CW_c[e] = C_w @ m
        CW = PolynomialMatrix(CW_c, n_r, (12,12))
        
        LMI2_c = {}
        for e in keys:
            W_blk = W_poly.coeffs.get(e, np.zeros((12,12)))
            CW_blk = CW.coeffs.get(e, np.zeros((12,12)))
            r1 = cp.hstack([lam*W_blk, np.zeros((12,6)), CW_blk.T])
            if sum(e)==0:
                r2 = cp.hstack([np.zeros((6,12)), (alpha-mu)*np.eye(6), np.zeros((6,12))])
                r3 = cp.hstack([CW_blk, np.zeros((12,6)), alpha*np.eye(12)])
            else:
                r2 = cp.hstack([np.zeros((6,30))])
                r3 = cp.hstack([CW_blk, np.zeros((12,18))])
            LMI2_c[e] = cp.vstack([r1,r2,r3])
        LMI2 = PolynomialMatrix(LMI2_c, n_r, (30,30))
        
        # Enforce SOS: (LMI2 - S2) >> 0
        constraints += sos_constraint(LMI2 - S2_term, degree)

    # 6. SOLVE
    print("Solving SOS...")
    prob = cp.Problem(cp.Minimize(alpha), constraints)
    try: prob.solve(solver=cp.MOSEK, verbose=True)
    except: prob.solve(solver=cp.SCS, verbose=True)

    if prob.status in ['optimal', 'optimal_inaccurate']:
        print(f"SUCCESS. Alpha={alpha.value:.4f}")
        
        # Standardize Output for Analyzer
        s_keys = sorted(basis_exps)
        W_out = [W_coeffs[k].value for k in s_keys]
        Y_out = [Y_coeffs[k].value for k in s_keys]
        
        return W_out, Y_out, s_keys
    else:
        print("Infeasible"); return None, None, None

# ==============================================================================
# PART 3: UNIVERSAL ANALYZER
# ==============================================================================
def analyze_results(Ws, Ys, basis_exps):
    """Works for both Rotation and Quaternion results."""
    dim_var = len(basis_exps[0])
    mode = 'ROTATION' if dim_var == 9 else 'QUATERNION'
    print(f"\n>>> ANALYZING {mode} CONTROLLER")
    
    # Symbols
    if mode == 'ROTATION': syms = sp.symbols('r11 r12 r13 r21 r22 r23 r31 r32 r33')
    else: syms = sp.symbols('qw qx qy qz')
        
    # Numerical Gain
    def get_K(vec):
        weights = []
        for exp in basis_exps:
            v = 1.0
            for i, p in enumerate(exp): 
                if p>0: v *= vec[i]**p
            weights.append(v)
        W = sum(Ws[k] * weights[k] for k in range(len(Ws)))
        Y = sum(Ys[k] * weights[k] for k in range(len(Ys)))
        return Y @ np.linalg.inv(W + 1e-6*np.eye(12))

    print("\nGain Scheduling Check:")
    if mode == 'ROTATION':
        K1 = get_K(np.eye(3).flatten())
        K2 = get_K(Rotation.from_euler('x', 45, degrees=True).as_matrix().flatten())
    else:
        K1 = get_K(np.array([1,0,0,0]))
        K2 = get_K(Rotation.from_euler('x', 45, degrees=True).as_quat()[[3,0,1,2]]) # [w,x,y,z]

    print(f"Hover K_px: {K1[0,0]:.4f}")
    print(f"Roll45 K_px: {K2[0,0]:.4f}")
    print(f"Norm Diff: {np.linalg.norm(K1-K2):.4f}")

# ==============================================================================
# SOLVER 1: ROTATION MATRIX (Tensorized, Degree 4)
# ==============================================================================
def solve_rccm_rotation_matrix(degree=4, num_samples=30):
    print(f"\n>>> SETUP: Rotation Matrix RCCM (Degree {degree}, Tensorized)...")
    
    n_x, n_u, n_r = 12, 6, 9
    basis_exps = get_monomial_basis(n_r, degree)
    n_basis = len(basis_exps)
    print(f"Basis Terms: {n_basis}")

    # Variables (Tensorized for speed)
    n_unique_W = (n_x * (n_x + 1)) // 2
    W_coeffs_flat = cp.Variable((n_unique_W, n_basis))
    Y_coeffs_flat = cp.Variable((n_u * n_x, n_basis))
    alpha = cp.Variable(nonneg=True)
    mu = cp.Variable(nonneg=True)

    # Constants & Scatter Matrix
    m, lam = 1.0, 1.0
    J_inv = np.linalg.inv(np.diag([0.02, 0.02, 0.04]))
    B_np = np.zeros((12,6)); B_np[6:9,0:3]=1/m*np.eye(3); B_np[9:12,3:6]=J_inv
    Bw_np = B_np.copy()
    
    # Scatter Matrix (C-Order for flattening)
    r_idx, c_idx = np.triu_indices(n_x)
    S_W = np.zeros((n_x*n_x, n_unique_W))
    for k, (r, c) in enumerate(zip(r_idx, c_idx)):
        S_W[r*n_x + c, k] = 1.0
        S_W[c*n_x + r, k] = 1.0

    # Weights
    C_w = block_diag(10*np.eye(3), 5*np.eye(3), 1*np.eye(3), 0.1*np.eye(3))

    # Sampling
    R_samples = [np.eye(3)]
    for _ in range(num_samples): R_samples.append(Rotation.random().as_matrix())
    
    w_corners = [np.zeros(3)]
    for c in itertools.product([-5, 5], repeat=3): w_corners.append(np.array(c))

    # Constraints
    constraints = []
    
    # 1. W0 >> 0
    w0_full = cp.reshape(S_W @ W_coeffs_flat[:,0], (12,12), order='C')
    constraints.append(w0_full >> 0.01 * np.eye(12))

    # Helper
    def get_phi(R_val, w_val):
        r = R_val.flatten(); w_hat = np.array([[0,-w_val[2],w_val[1]],[w_val[2],0,-w_val[0]],[-w_val[1],w_val[0],0]])
        rd = (R_val @ w_hat).flatten()
        phi, phid = np.zeros(n_basis), np.zeros(n_basis)
        for k, e in enumerate(basis_exps):
            v=1.0; 
            for i,p in enumerate(e): 
                if p>0: v*=r[i]**p
            phi[k]=v
            if sum(e)==0: continue
            d=0.0
            for i,p in enumerate(e):
                if p>0:
                    term = p*(r[i]**(p-1))*rd[i]
                    rest=1.0
                    for j,pj in enumerate(e): 
                        if i!=j and pj>0: rest*=r[j]**pj
                    d+=term*rest
            phid[k]=d
        return phi, phid

    print(f"Building constraints for {len(R_samples)} orientations...")
    for R_val in R_samples:
        # W(R) >> 0
        p, _ = get_phi(R_val, np.zeros(3))
        w_R = cp.reshape(S_W @ (W_coeffs_flat @ p), (12,12), order='C')
        constraints.append(w_R >> 0.01 * np.eye(12))
        
        for w_val in w_corners:
            p, pd = get_phi(R_val, w_val)
            
            W = cp.reshape(S_W @ (W_coeffs_flat @ p), (12,12), order='C')
            Y = cp.reshape(Y_coeffs_flat @ p, (6,12), order='C')
            Wd = cp.reshape(S_W @ (W_coeffs_flat @ pd), (12,12), order='C')
            
            # Dynamics
            w_hat = np.array([[0,-w_val[2],w_val[1]],[w_val[2],0,-w_val[0]],[-w_val[1],w_val[0],0]])
            Jw = np.diag([0.02,0.02,0.04]) @ w_val
            Jw_hat = np.array([[0,-Jw[2],Jw[1]],[Jw[2],0,-Jw[0]],[-Jw[1],Jw[0],0]])
            term44 = J_inv @ (Jw_hat - w_hat @ np.diag([0.02,0.02,0.04]))
            
            A = np.zeros((12,12)); A[0:3,6:9]=np.eye(3); A[3:6,3:6]=-w_hat; A[3:6,9:12]=np.eye(3); A[9:12,9:12]=term44
            
            # LMI 1 (Stability)
            He = (A@W + B_np@Y) + (A@W + B_np@Y).T
            Blk1 = -Wd + He + 2*lam*W
            row1 = cp.hstack([Blk1, Bw_np])
            row2 = cp.hstack([Bw_np.T, -mu*np.eye(6)])
            constraints.append(cp.vstack([row1, row2]) << 0)
            
            # LMI 2 (Tube)
            CW = C_w @ W
            r1 = cp.hstack([lam*W, np.zeros((12,6)), CW.T])
            r2 = cp.hstack([np.zeros((6,12)), (alpha-mu)*np.eye(6), np.zeros((6,12))])
            r3 = cp.hstack([CW, np.zeros((12,6)), alpha*np.eye(12)])
            constraints.append(cp.vstack([r1,r2,r3]) >> 0)

    print("Solving (SCS)...")
    prob = cp.Problem(cp.Minimize(alpha), constraints)
    try: prob.solve(solver=cp.SCS, verbose=True, eps=1e-3)
    except: prob.solve(solver=cp.MOSEK, verbose=True)

    if prob.status in ['optimal', 'optimal_inaccurate']:
        print(f"SUCCESS. Alpha={alpha.value:.4f}")
        # Unpack Tensor
        Wf = W_coeffs_flat.value
        Yf = Y_coeffs_flat.value
        W_list, Y_list = [], []
        for k in range(n_basis):
            W_list.append((S_W @ Wf[:,k]).reshape(12,12)) # numpy default C
            Y_list.append(Yf[:,k].reshape(6,12))
        return W_list, Y_list, basis_exps, alpha.value
    else:
        print("Infeasible"); return None, None, None, None

# ==============================================================================
# SOLVER 2: QUATERNION (List-based, Degree 2/4)
# ==============================================================================
def solve_rccm_quaternion(degree=2, num_samples=50):
    print(f"\n>>> SETUP: Quaternion RCCM (Degree {degree}, Sampled)...")
    
    n_x, n_u, n_q = 12, 6, 4
    basis_exps = get_monomial_basis(n_q, degree)
    n_basis = len(basis_exps)
    print(f"Basis Terms: {n_basis}")

    Ws = [cp.Variable((n_x, n_x), symmetric=True) for _ in range(n_basis)]
    Ys = [cp.Variable((n_u, n_x)) for _ in range(n_basis)]
    alpha = cp.Variable(nonneg=True)
    mu = cp.Variable(nonneg=True)

    m, lam = 1.0, 1.0
    J_inv = np.linalg.inv(np.diag([0.02, 0.02, 0.04]))
    B_np = np.zeros((12,6)); B_np[6:9,0:3]=1/m*np.eye(3); B_np[9:12,3:6]=J_inv
    Bw_np = B_np.copy()
    C_w = block_diag(10*np.eye(3), 5*np.eye(3), 1*np.eye(3), 0.1*np.eye(3))

    # Samples
    q_samples = [np.array([1,0,0,0])]
    for _ in range(num_samples):
        q = np.random.randn(4); q/=np.linalg.norm(q); q_samples.append(q)
    
    w_corners = [np.zeros(3)]
    for c in itertools.product([-5, 5], repeat=3): w_corners.append(np.array(c))

    constraints = []
    constraints.append(Ws[0] >> 0.01 * np.eye(12))

    print(f"Building constraints for {len(q_samples)} orientations...")
    
    for q_val in q_samples:
        # Precompute Basis
        phi = np.zeros(n_basis)
        for k, e in enumerate(basis_exps):
            v=1.0
            for i,p in enumerate(e): v*=q_val[i]**p
            phi[k] = v
            
        W_curr = cp.sum([Ws[k]*phi[k] for k in range(n_basis) if abs(phi[k])>1e-6])
        Y_curr = cp.sum([Ys[k]*phi[k] for k in range(n_basis) if abs(phi[k])>1e-6])
        constraints.append(W_curr >> 0.01*np.eye(12))

        for w_val in w_corners:
            # W_dot
            wx,wy,wz=w_val
            Omega = 0.5*np.array([[0,-wx,-wy,-wz],[wx,0,wz,-wy],[wy,-wz,0,wx],[wz,wy,-wx,0]])
            qd = Omega @ q_val
            
            phid = np.zeros(n_basis)
            for k, e in enumerate(basis_exps):
                if sum(e)==0: continue
                d=0.0
                for i,p in enumerate(e):
                    if p>0:
                        term=p*(q_val[i]**(p-1))*qd[i]
                        rest=1.0
                        for j,pj in enumerate(e): 
                            if i!=j: rest*=q_val[j]**pj
                        d+=term*rest
                phid[k]=d
            
            Wd_curr = cp.sum([Ws[k]*phid[k] for k in range(n_basis) if abs(phid[k])>1e-6])

            # Dynamics
            w_hat = np.array([[0,-w_val[2],w_val[1]],[w_val[2],0,-w_val[0]],[-w_val[1],w_val[0],0]])
            Jw = np.diag([0.02,0.02,0.04]) @ w_val
            Jw_hat = np.array([[0,-Jw[2],Jw[1]],[Jw[2],0,-Jw[0]],[-Jw[1],Jw[0],0]])
            term44 = J_inv @ (Jw_hat - w_hat @ np.diag([0.02,0.02,0.04]))
            
            A = np.zeros((12,12)); A[0:3,6:9]=np.eye(3); A[3:6,3:6]=-w_hat; A[3:6,9:12]=np.eye(3); A[9:12,9:12]=term44

            # LMI 1
            He = (A@W_curr + B_np@Y_curr) + (A@W_curr + B_np@Y_curr).T
            Blk1 = -Wd_curr + He + 2*lam*W_curr
            r1 = cp.hstack([Blk1, Bw_np])
            r2 = cp.hstack([Bw_np.T, -mu*np.eye(6)])
            constraints.append(cp.vstack([r1,r2]) << 0)

            # LMI 2
            CW = C_w @ W_curr
            r1 = cp.hstack([lam*W_curr, np.zeros((12,6)), CW.T])
            r2 = cp.hstack([np.zeros((6,12)), (alpha-mu)*np.eye(6), np.zeros((6,12))])
            r3 = cp.hstack([CW, np.zeros((12,6)), alpha*np.eye(12)])
            constraints.append(cp.vstack([r1,r2,r3]) >> 0)

    print("Solving (SCS)...")
    prob = cp.Problem(cp.Minimize(alpha), constraints)
    try: prob.solve(solver=cp.SCS, verbose=True, eps=1e-3)
    except: prob.solve(solver=cp.MOSEK, verbose=True, eps=1e-3)

    if prob.status in ['optimal', 'optimal_inaccurate']:
        print(f"SUCCESS. Alpha={alpha.value:.4f}")
        W_vals = [m.value for m in Ws]
        Y_vals = [m.value for m in Ys]
        return W_vals, Y_vals, basis_exps, alpha.value
    else:
        print("Infeasible"); return None, None, None, None

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # SELECT MODE HERE
    MODE = 'QUATERNION' # or 'ROTATION'
    DEGREE = 2          # 2 or 4
    
    # sample-based
    # if MODE == 'ROTATION':
    #     Ws, Ys, exps, alpha = solve_rccm_rotation_matrix(degree=DEGREE, num_samples=30)
    # else:
    #     Ws, Ys, exps, alpha = solve_rccm_quaternion(degree=DEGREE, num_samples=50)
    
    # full SOS
    Ws, Ys, exps = solve_rccm_sos(mode='QUATERNION', degree=4)
        
    if Ws is not None:
        # 1. Analyze
        analyze_results(Ws, Ys, exps)
        
        # 2. Save for Simulator
        data = {
            "W_matrices": Ws,
            "Y_matrices": Ys,
            "basis_exponents": exps,
            "alpha": alpha,
            "mode": MODE,
            "degree": DEGREE
        }
        filename = f"rccm_data_{MODE.lower()}_deg{DEGREE}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"\nSaved controller to {filename}")
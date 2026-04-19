## SADF -- Simplified Affine Disturbance Feedback 

## This code is working and it saves the data and plots also

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from datetime import datetime
import time
from scipy.linalg import solve_discrete_are


## to save the data and plot make    ------------------------->   save_fig = True

save_fig = True


x_real = np.array([0, 0, 0, 0]) 
x_targ = np.array([3.0, 2, 0, 0]) 

nx = 4           # State: [x, y, vx, vy]
nu = 2           # Control: [ax, ay]
nw = 2 
T_sim = 50      
K = 8           # Prediction Horizon
L = 10           # Number of samples
       

beta = 0.01      
epsilon = 0.001
dmin = 0.1      


dt = 0.1 
A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
B = np.array([
    [0.5*dt**2, 0],
    [0, 0.5*dt**2],
    [dt, 0],
    [0, dt]
])
C_out = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])


umin = -5
umax = 5
xmin = np.array([-0.1, -0.1])
xmax = np.array([3.5, 3])

# Uncertainty Supports
d_bound = 0.05
H_d = np.vstack([np.eye(nx), -np.eye(nx)])
h_d = d_bound * np.ones(2*nx)

w_bound = 0.05
H_w = np.vstack([np.eye(nw), -np.eye(nw)])
h_w = w_bound * np.ones(2*nw)

# np.random.seed(42)
d_samples = np.random.uniform(-d_bound, d_bound, (nx, K, L))
w_samples = np.random.uniform(-w_bound, w_bound, (nw, K, L))

# Obstacle

# obs_centers = [[1.5, 1.0],[1.5, 2.5],[2.8, 0.5]]
# num_obs = len(obs_centers)
# radii = [0.5, 0.5, 0.3]
# poly_sides_n = [6, 6, 4]  

obs_centers = [[1.5, 0.5], [1.5, 2]]
num_obs = len(obs_centers)
radii = [0.3, 0.3]
poly_sides_n = [6, 6]

# obs_centers = [[1.5, 0.5]]
# num_obs = len(obs_centers)
# radii = [0.5]
# poly_sides_n = [6]

def lifted_matrices(A, B, k):
    Ak = np.zeros((nx, nx + k*nu))
    Dk = np.zeros((nx, k*nx))
    Ak[:, :nx] = np.linalg.matrix_power(A, k)
    for i in range(k):
        Apow = np.linalg.matrix_power(A, k-1-i)  
        Ak[:, nx + i*nu : nx + (i+1)*nu] = Apow @ B
        Dk[:, i*nx:(i+1)*nx] = Apow
    return Ak, Dk


# Opti formulation
opti = ca.Opti()

X_sym = ca.MX.sym('x', nx, K+1)
Pi_sym = ca.MX.sym('pi', nu, K) 
X = opti.variable(X_sym)
Pi = opti.variable(Pi_sym)


M_vars = [opti.variable(nu, nx) for _ in range(K-1)]

eta = opti.variable(K)
lam = opti.variable(K)
s   = opti.variable(K, L)

x0_param    = opti.parameter(nx)
xref_param  = opti.parameter(nx)
u_prev_param = opti.parameter(nu)

# Q = np.diag([4500, 4500, 1500, 1500])
Q = np.diag([2000, 2000, 800, 800])
R = 0.5 * np.eye(nu)
S = 50.0 * np.eye(nu)
P = solve_discrete_are(A,B,Q,R)

cost = 0
cost += ca.mtimes([(X[:,K]-xref_param).T, P, (X[:,K]-xref_param)]) 

for k in range(K):
    cost += ca.mtimes([(X[:,k]-xref_param).T, Q, (X[:,k]-xref_param)]) 
    cost += ca.mtimes([Pi[:,k].T, R, Pi[:,k]]) 

    if k == 0:
        dpi = Pi[:,k] - u_prev_param
    else:
        dpi = Pi[:,k] - Pi[:,k-1]
    cost += ca.mtimes([dpi.T, S, dpi])


for m in M_vars:
    cost += 10.0 * ca.sumsqr(m)
    # opti.subject_to(opti.bounded(-5.0, ca.vec(m), 5.0))


opti.minimize(cost)


# Constraints

nu_vars = []
gam_vars = []
opti.set_initial(lam, 1.0)
opti.set_initial(s, 0.1)
opti.set_initial(eta, 0.0)
opti.subject_to(X[:,0] == x0_param)


for k in range(K):
    opti.subject_to(X[:,k+1] == A @ X[:,k] + B @ Pi[:,k])
    opti.subject_to(opti.bounded(umin, Pi[:,k], umax))
    opti.subject_to(X[0,k] >= xmin[0])
    opti.subject_to(X[0,k] <= xmax[0])
    opti.subject_to(X[1,k] <= xmax[1])
    opti.subject_to(X[1,k] >= xmin[1])


## Risk term 
for k in range(1, K+1):
    risk_term = eta[k-1] + (1/beta)*(lam[k-1]*epsilon + (1/L)*ca.sum1(s[k-1,:]))
    opti.subject_to(risk_term <= 0)
    opti.subject_to(lam[k-1] >= 1e-4)


## Function for DRCVaR collision avoidance constraint
def DRCVaR(obs_center, E, e, Mk_set, lifted_set, Hh_joint_set):
    
    no = E.shape[0]

    for k in range(1, K+1):

        Ak, Dk, Bk = lifted_set[k] 
        Ak_hat = E @ C_out @ Ak
        Dk_hat = E @ C_out @ Dk
        
        M_k = Mk_set[k]
        
        
        B_bar = E @ C_out @ Bk @ M_k
        Dk_bar = B_bar + Dk_hat


        H_joint, h_joint, n_gam = Hh_joint_set[k]

        nu_L  = opti.variable(no, L)
        gam_L = opti.variable(n_gam, L)
        
        nu_vars.append(nu_L)
        gam_vars.append(gam_L)
        

        opti.set_initial(nu_L, 0.1)
        opti.set_initial(gam_L, 0.1)


        M_xi = ca.horzcat( Dk_bar, -E )

        Pi_seq = ca.vertcat(*[Pi[:,i] for i in range(k)])
        P1 = ca.vertcat(x0_param, Pi_seq) 
        Q1 = Ak_hat @ P1 - e

        for l in range(L):
            nu_l = nu_L[:, l]
            gam_l = gam_L[:, l]
            
            # Sample xi^(l)
            d_seq_l = d_samples[:, 0:k, l].flatten('F')
            w_k_l   = w_samples[:, k-1, l]
            # xi_hat  = ca.vertcat(d_seq_l, w_k_l)
            xi_hat = np.concatenate([d_seq_l, w_k_l])
                    
            T1 = ca.mtimes(Q1.T, nu_l)

            P2 = ca.mtimes(M_xi.T, nu_l) + ca.mtimes(H_joint.T, gam_l)
            T2 = ca.mtimes(P2.T, xi_hat)
            
            T3 = ca.mtimes(gam_l.T, h_joint)

            opti.subject_to(dmin - (T1 + T2 - T3) <= s[k-1, l] + eta[k-1])

            opti.subject_to(ca.sumsqr(P2) <= lam[k-1]**2)
            opti.subject_to(ca.sumsqr(ca.mtimes(E.T, nu_l)) <= 1)

            opti.subject_to(nu_l >= 0)
            opti.subject_to(gam_l >= 0)
            opti.subject_to(s[k-1, l] >= 0)


E = []
e = []
# Change the support set to change the level of uncertainty for each obs
H_w = np.vstack([np.eye(nw), -np.eye(nw)])
h_w = w_bound * np.ones(2*nw)

## Mk|t
Mk_set = {}
lifted_set = {}
Hh_joint_set = {}

for k in range(1, K+1):

    Ak, Dk = lifted_matrices(A, B, k)
    Bk     = Ak[:, nx:]
    lifted_set[k] = (Ak, Dk, Bk)

    if k == 1:
        Mk_set[k] = ca.MX.zeros(nu, nx)
    else:
        rows = []
        for i in range(k):
            cols = [M_vars[i-j-1] if i > j else ca.MX.zeros(nu, nx)
                    for j in range(k)]
            rows.append(ca.horzcat(*cols))
        Mk_set[k] = ca.vertcat(*rows)

    H_d_batch = np.kron(np.eye(k), H_d)              
    h_d_batch = np.kron(np.ones(k), h_d)           
    nH_d, nH_w = H_d_batch.shape[0], H_w.shape[0]
    H_joint_np = np.zeros((nH_d + nH_w, nx*k + nw))   
    H_joint_np[:nH_d, :nx*k] = H_d_batch
    H_joint_np[nH_d:, nx*k:] = H_w
    h_joint_np = np.concatenate([h_d_batch, h_w])      
    n_gam      = nH_d + nH_w                        
    Hh_joint_set[k] = (H_joint_np, h_joint_np, n_gam)


for n in range(num_obs):
    obs_center = obs_centers[n]
    angle = np.linspace(0, 2*np.pi, poly_sides_n[n]+1)[:-1]
    normals = np.column_stack((np.cos(angle), np.sin(angle)))
    E.append(normals)
    e.append(np.sum(normals * (obs_center + radii[n] * normals), axis=1) )
    
    DRCVaR(obs_center, E[n], e[n], Mk_set, lifted_set, Hh_joint_set)



# Solver 
p_opts = {'expand': True,       
          'print_time' : 0}
s_opts = {
    'linear_solver': 'ma97',
    'hsllib': '/home/trees/scratch/hsl-bin/lib/libcoinhsl.so',
    'max_iter': 1000,
    'print_level': 0,
    'sb': 'yes', 
    'tol': 1e-4,
    'acceptable_tol': 1e-3,     
    'constr_viol_tol': 1e-4
}
opti.solver('ipopt', p_opts, s_opts)

# Simulation 



u_last = np.zeros(nu)

hist_x = [x_real]
hist_u = []
sol_x_pred = []

last_X_sol = None
last_Pi_sol = None
last_duals = None

solve_times = []
distances_to_obstacle = []
step_indices = []

collision_count = 0
failed_count = 0
collision_flag = False

for t in range(T_sim):

    # d_samples = np.random.uniform(-d_bound, d_bound, (nx, K, L))
    # w_samples = np.random.uniform(-w_bound, w_bound, (nw, K, L))


    # Changing obs centre location at each time step
    # obs_centers.append(obs_center)
    # if t<=30:
    #     obs_center = np.array([obs_center[0] + 0.005 * t, obs_center[1] + 0.005 * t])

    # checking for collision 
    for n in range(num_obs):    
        if np.all(E[n] @ x_real[0:2] <= e[n]):
            collision_count += 1
            print("Collision !!!")
            collision_flag = True
            break

    if collision_flag:
        break

    dist = np.linalg.norm(x_real[:2] - x_targ[:2])
    if dist <= 0.1:
        print("Reached !!!!")
        break

    step_indices.append(t)
    opti.set_value(x0_param, x_real)
    opti.set_value(xref_param, x_targ)
    opti.set_value(u_prev_param, u_last)
    
    
    if last_X_sol is not None:
        opti.set_initial(X,  ca.horzcat(last_X_sol[:, 1:], last_X_sol[:, -1]))
        opti.set_initial(Pi, ca.horzcat(last_Pi_sol[:, 1:], last_Pi_sol[:, -1]))
        if last_duals is not None:
            lam_v, eta_v, s_v, mv_list, nu_list, gam_list = last_duals
            opti.set_initial(lam, lam_v)
            opti.set_initial(eta, eta_v)
            opti.set_initial(s,   s_v)
            for m_v, val in zip(M_vars,   mv_list):  opti.set_initial(m_v, val)
            for nv,  val in zip(nu_vars,  nu_list):   opti.set_initial(nv,  val)
            for gv,  val in zip(gam_vars, gam_list):  opti.set_initial(gv,  val)
    else:
        # opti.set_initial(X,   ca.repmat(x_real, 1, K+1))
        # opti.set_initial(Pi,  np.zeros((nu, K)))
        # straight-line interpolation from x_real to x_targ
        X_init = np.zeros((nx, K+1))
        for i in range(K+1):
            X_init[:, i] = x_real + (x_targ - x_real) * i / K
        opti.set_initial(X, X_init)
        # constant control pointing toward target
        dx      = x_targ - x_real
        u_hint  = np.clip(dx[:nu] / (K * dt), umin, umax)
        opti.set_initial(Pi, np.tile(u_hint.reshape(-1,1), (1, K)))
        opti.set_initial(lam, 1.0)
        opti.set_initial(s,   0.1)
        opti.set_initial(eta, 0.0)
        for m_v in M_vars:   opti.set_initial(m_v, np.zeros((nu, nx)))
        for nv in nu_vars:   opti.set_initial(nv,  0.1)
        for gv in gam_vars:  opti.set_initial(gv,  0.1)
    start_time = time.time()

    try:
        sol = opti.solve()
        stats = sol.stats()
        # print(f"At time {t}:")
        # print(f"  Status: {stats['return_status']}")
        # print(f"  Iterations: {stats['iter_count']}")
        # print(f"  Constraint violations: {stats.get('constr_viol_max', 'N/A')}")
        solve_duration = time.time() - start_time
        x_pred = sol.value(X)
        pi_opt = sol.value(Pi)
        
        u_apply = pi_opt[:, 0]
        
        print(f"At time {t} : {sol.stats()['return_status']}, Distance : {dist}")

        if sol.stats()['return_status'] != 'Solve_Succeeded':
            failed_count += 1
            break

        last_X_sol = x_pred
        last_Pi_sol = pi_opt
        last_duals  = (                                   
            sol.value(lam),
            sol.value(eta),
            sol.value(s),
            [sol.value(m_v) for m_v in M_vars],
            [sol.value(nv)  for nv  in nu_vars],
            [sol.value(gv)  for gv  in gam_vars],
        )

        # if dist < d_bound:
        #     u_apply = np.zeros(nu)
        #     last_X_sol = None
        #     last_duals = None

        
        sol_x_pred.append(x_pred)

    except Exception as ex:
        # if sol.stats()['return_status'] != 'Solve_Succeeded':
        #     failed_count += 1
        #     break
        solve_duration = time.time() - start_time
        print(f"At time {t} : Exception occurred, Distance : {dist}")
        u_apply = np.zeros(nu)
        last_X_sol = None
        last_duals = None

    solve_times.append(solve_duration)
        
    d_real = np.random.uniform(-d_bound, d_bound, nx)
    x_next = A @ x_real + B @ u_apply + d_real
    x_real = x_next
    u_last = u_apply
    # x_targ = x_pred[:, -1] if last_X_sol is not None else x_targ
    
    hist_x.append(x_real)
    hist_u.append(u_apply)



hist_x = np.array(hist_x).T
hist_u = np.array(hist_u).T


## PLotting results 


plt.figure(figsize=(10, 8))

from matplotlib.patches import Rectangle
rect = Rectangle(
    (xmin[0], xmin[1]),           # Bottom-left corner
    xmax[0] - xmin[0],            # Width
    xmax[1] - xmin[1],            # Height
    linewidth=2.5,
    edgecolor='black',
    facecolor='none',
    linestyle='--',
    label='Boundaries'
)
plt.gca().add_patch(rect)

for n in range(num_obs):
    np.random.seed(99)
    for _ in range(50):
        w_sample = np.random.uniform(-w_bound, w_bound, 2)
        noisy_verts = (obs_centers[n] + w_sample) + radii[n] * normals
        noisy_patch = Polygon(noisy_verts, closed=True, color='red', alpha=0.05, edgecolor=None)
        plt.gca().add_patch(noisy_patch)

    
    nom_verts = obs_centers[n] + radii[n] * normals
    nom_patch = Polygon(nom_verts, closed=True, fill=False, edgecolor='darkred', linewidth=2)#, label='Nominal')
    plt.gca().add_patch(nom_patch)


    margin_verts = obs_centers[n] + (radii[n] + dmin) * normals
    margin_patch = Polygon(margin_verts, closed=True, 
                        fill=False, linestyle='--', color='darkred', alpha=0.8, label='Safety Margin')
    plt.gca().add_patch(margin_patch)



plt.plot(hist_x[0, :], hist_x[1, :], 'b-', linewidth=2, label='Trajectory')
plt.plot(hist_x[0, 0], hist_x[1, 0], 'go', markersize=10, label='Start')
plt.plot(x_targ[0], x_targ[1], 'g*', markersize=15, label='Target')

for i, pred in enumerate(sol_x_pred):
    label = "MPC Prediction" if i == 0 else None
    plt.plot(pred[0, :], pred[1, :], 'c--', alpha=0.5, label=label)
    plt.scatter(pred[0,-1], pred[1,-1])

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title(f'DR-MPC (T={T_sim})')
plt.legend(loc='best')
plt.grid(True)
# plt.axis('equal')
plt.xlim(xmin[0]-0.2, xmax[0]+0.1)
plt.ylim(xmin[1]-0.2, xmax[1]+0.1)

from pathlib import Path
print(Path.cwd())

if save_fig:
    now = datetime.now().strftime("%H%M_%d%m%y")
    plt.savefig(f'SADF_{now}_Traj_C.png', dpi=600, bbox_inches='tight')
plt.show()

# Controls
plt.figure(figsize=(10, 4))
plt.step(range(len(hist_u[0])), hist_u[0], label='ax')
plt.step(range(len(hist_u[1])), hist_u[1], label='ay')
plt.ylabel('Acceleration [m/s^2]')
plt.xlabel('Time Step')
plt.title('Control Inputs')
plt.legend()
plt.grid(True)

if save_fig:
    plt.savefig(f'SADF_{now}__Controls_C.png', dpi=600, bbox_inches='tight')
plt.show()


# Velocities
plt.figure(figsize=(10, 4))
plt.step(range(len(hist_x[2])), hist_x[2], label='vx')
plt.step(range(len(hist_x[3])), hist_x[3], label='vy')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time Step')
plt.title('Velocities')
plt.legend()
plt.grid(True)

if save_fig:
    plt.savefig(f'SADF_{now}__Velocities_C.png', dpi=600, bbox_inches='tight')
plt.show()
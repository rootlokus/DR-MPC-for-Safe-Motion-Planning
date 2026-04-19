import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from datetime import datetime
import time
from scipy.linalg import solve_discrete_are
import pandas as pd
import gc



start = [0,0]
target = [3,2]
K = 8
L = 15
epsilon = 0.0001
beta = 0.01
nx = 4           # State: [x, y, vx, vy]
nu = 2           # Control: [ax, ay]
nw = 2 

d_bound = 0.06  # plant uncertainty support bound 
w_bound = 0.06  # obstacle pos uncertainty support bound

d_samples = np.random.uniform(-d_bound, d_bound, (nx, K, L))
w_samples = np.random.uniform(-w_bound, w_bound, (nw, K, L))


x_real  = np.array([start[0], start[1], 0, 0])
x_targ  = np.array([target[0], target[1], 0, 0])
nx,nu = 4, 2
T_sim = 150
K_hor = 3         
dt    = 0.1
A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1,  0],
    [0, 0, 0,  1]
])
B = np.array([
    [0.5*dt**2, 0],
    [0, 0.5*dt**2],
    [dt, 0],
    [0,  dt]
])
umin, umax = -5.0, 5.0
xmin = np.array([-0.5, -0.5])   
xmax = np.array([ 3.5,  3.0])   
dmin    = 0.1   

obs_centers = [[1.5, 0.5], [1.5, 2]]
num_obs = len(obs_centers)
radii = [0.3, 0.3]
poly_sides_n = [6, 6]



Q_tube = np.diag([10.0, 10.0, 1.0, 1.0])
R_tube = 5.0 * np.eye(nu) 
P_tube = solve_discrete_are(A, B, Q_tube, R_tube)
K_tube = -np.linalg.lstsq(R_tube + B.T @ P_tube @ B, B.T @ P_tube @ A, rcond=None)[0]                                    
A_cl = A + B @ K_tube  
# print("Eigenvalues of A_cl:", np.linalg.eigvals(A_cl))

Q_mpc = np.diag([2000.0, 2000.0, 800.0, 800.0])
R_mpc = 0.5 * np.eye(nu)
S_cost = 50.0 * np.eye(nu)
P_mpc = solve_discrete_are(A, B, Q_mpc, R_mpc)
e_max = np.array([d_bound, d_bound, 0.0, 0.0])

def propagate_tube(A_cl, e_max, N):
    alpha = np.zeros((N+1, nx))
    for k in range(1, N+1):
        bound = np.zeros(nx)
        A_pow = np.eye(nx)
        for j in range(k):
            bound += np.abs(A_pow) @ e_max
            A_pow = A_cl @ A_pow
        alpha[k] = bound
    return alpha
alpha = propagate_tube(A_cl, e_max, K_hor)

# print("Offline Tube Half-Widths (alpha_k):")
# for k in range(K_hor+1):
#     print(f"  k={k}: {alpha[k].round(4)}")


# Obstacle 
E_obs, e_obs_nom = [], []
for n in range(num_obs):
    angle   = np.linspace(0, 2*np.pi, poly_sides_n[n]+1)[:-1]
    normals = np.column_stack((np.cos(angle), np.sin(angle)))  
    oc      = np.array(obs_centers[n])
    E_obs.append(normals)
    e_obs_nom.append(np.array([
        normals[m] @ (oc + radii[n]*normals[m]) for m in range(len(normals))
    ]))


def build_robust_mpc(x0_val, x_targ, u_prev):
    opti = ca.Opti()
    Z = opti.variable(nx, K_hor+1)
    V = opti.variable(nu, K_hor)
    x0_p    = opti.parameter(nx)
    xref_p  = opti.parameter(nx)
    uprev_p = opti.parameter(nu)
    
    cost = ca.mtimes([(Z[:,K_hor]-xref_p).T, P_mpc, (Z[:,K_hor]-xref_p)])
    for k in range(K_hor):
        cost += ca.mtimes([(Z[:,k]-xref_p).T, Q_mpc, (Z[:,k]-xref_p)])
        cost += ca.mtimes([V[:,k].T, ca.DM(R_mpc), V[:,k]])
        dv    = V[:,k] - (uprev_p if k==0 else V[:,k-1])
        cost += ca.mtimes([dv.T, ca.DM(S_cost), dv])
    opti.minimize(cost)
   
    opti.subject_to(Z[:,0] == x0_p)
    for k in range(K_hor):
        opti.subject_to(Z[:,k+1] == ca.DM(A) @ Z[:,k] + ca.DM(B) @ V[:,k])
    for k in range(1, K_hor+1): 
        ak = alpha[k]
        
        opti.subject_to(Z[0,k] >= xmin[0] + ak[0])
        opti.subject_to(Z[0,k] <= xmax[0] - ak[0])
        opti.subject_to(Z[1,k] >= xmin[1] + ak[1])
        opti.subject_to(Z[1,k] <= xmax[1] - ak[1])
        
        if k < K_hor:
            u_tight = np.abs(K_tube) @ ak   
            opti.subject_to(V[:,k] >= umin + u_tight)
            opti.subject_to(V[:,k] <= umax - u_tight)
    beta = 15.0 
    for n in range(num_obs):
        normals = E_obs[n]   
        e_nom_n = e_obs_nom[n]
        for k in range(1, K_hor+1):
            ak = alpha[k]
            face_distances = []
            
            for m in range(normals.shape[0]):
                nx_m, ny_m = normals[m, 0], normals[m, 1]
                
                gamma_km = (abs(nx_m) * ak[0] + abs(ny_m) * ak[1]) + \
                        (abs(nx_m) * w_bound + abs(ny_m) * w_bound)
                
                dist_m = nx_m * Z[0,k] + ny_m * Z[1,k] - (e_nom_n[m] + dmin + gamma_km)
                face_distances.append(dist_m)
            
            sum_exp = ca.sum1(ca.exp(beta * ca.vertcat(*face_distances)))
            smooth_max = ca.log(sum_exp) / beta
            opti.subject_to(smooth_max >= 0)

    opti.set_value(x0_p,    x0_val)
    opti.set_value(xref_p,  x_targ)
    opti.set_value(uprev_p, u_prev)
    
    opti.set_initial(Z, ca.repmat(ca.DM(x0_val), 1, K_hor+1))
    opti.set_initial(V, 0)
    
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
    try:
        sol  = opti.solve()
        return sol.value(V[:,0]), sol.value(Z), True, sol.value(cost)
    except Exception as ex:
        print(f" Solver failed.")
        return np.zeros(nu), opti.debug.value(Z), False, sol.debug.value(cost)



u_last  = np.zeros(nu)
hist_x      = [x_real.copy()]
hist_u      = []
solve_times = []
costs = []
sol_z_pred  = []
failed_count = 0
failed_flag = True
collision_count = 0
experienced_infeasibility = False
infeasible_but_reached_count = 0

for t in range(T_sim):
    
    dist = np.linalg.norm(x_real[:2] - x_targ[:2])
    if dist <= 0.15:
        print(f"  t={t}: REACHED !! dist={dist:.4f}")
        failed_flag = False
        
        if experienced_infeasibility:
            print("  Reached target despite solver failure!")
            infeasible_but_reached_count = 1
            
        break
    # Check for collision
    collision_flag = False
    for n in range(num_obs):
        if np.all(E_obs[n] @ x_real[0:2] <= e_obs_nom[n]):
            collision_count += 1
            print(f"  t={t}: COLLISION !!")
            collision_flag = True
            break
    
    if collision_flag:
        break
    start_time = time.time()
    v_opt, Z_pred, ok, cost = build_robust_mpc(x_real, x_targ, u_last)
    solve_duration = time.time() - start_time
    solve_times.append(solve_duration)
    sol_z_pred.append(Z_pred)
    costs.append(cost)
    if not ok:
        print(f"  t={t}: INFEASIBLE.")
        experienced_infeasibility = True
        u_apply = u_last  # Apply the last known safe input
    else:
        u_apply = v_opt
    
    d_real = np.zeros(nx)
    d_bound1 = 0.08
    d_real[:2] = np.random.uniform(-d_bound1, d_bound1, 2)
    
    x_real = A @ x_real + B @ u_apply + d_real
    u_last = u_apply
    hist_x.append(x_real.copy())
    hist_u.append(u_apply.copy())
    print(f"  t={t:2d}  | time={solve_duration:.2f}s | dist={dist:.4f} | u=({u_apply[0]:.2f},{u_apply[1]:.2f})")


hist_x = np.array(hist_x).T
hist_u = np.array(hist_u).T

if failed_flag:
    failed_count += 1



#  Plotting results
from matplotlib.patches import Polygon, Rectangle

fig, ax = plt.subplots(figsize=(10, 8))
rect = Rectangle((xmin[0], xmin[1]), xmax[0]-xmin[0], xmax[1]-xmin[1],
                  linewidth=2.5, edgecolor='black', facecolor='none',
                  linestyle='--', label='Boundaries')
ax.add_patch(rect)

# Obstacles
normals_plot6 = np.column_stack((
    np.cos(np.linspace(0, 2*np.pi, 7)[:-1]),
    np.sin(np.linspace(0, 2*np.pi, 7)[:-1])
))
rng99 = np.random.default_rng(99)
for obs_idx in range(num_obs):
    oc  = np.array(obs_centers[obs_idx])
    nrm = E_obs[obs_idx]
    for _ in range(50):
        w_s = rng99.uniform(-d_bound, d_bound, 2)
        ax.add_patch(Polygon((oc+w_s) + radii[obs_idx]*nrm,
                              closed=True, color='red', alpha=0.05))
    ax.add_patch(Polygon(oc + radii[obs_idx]*nrm,
                         closed=True, fill=False,
                         edgecolor='darkred', linewidth=2))
    ax.add_patch(Polygon(oc + (radii[obs_idx]+dmin)*nrm,
                         closed=True, fill=False, linestyle='--',
                         edgecolor='darkred', alpha=0.8,
                         label='Safety Margin' if obs_idx == 0 else None))


# Main trajectory
ax.plot(hist_x[0], hist_x[1], 'b-', linewidth=2.5, label='RMPC Trajectory')
ax.plot(hist_x[0,0], hist_x[1,0], 'go', ms=10, label='Start')
ax.plot(x_targ[0],  x_targ[1],   'g*', ms=15, label='Target')

# Predictions
for i, pred in enumerate(sol_z_pred):
    if pred is not None:
        ax.plot(pred[0], pred[1], 'c--', alpha=0.4,
                label='MPC Prediction' if i == 0 else None)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title(f'Robust MPC (T={t})')
ax.legend(loc='best')
ax.grid(True)
ax.set_xlim(xmin[0]-0.2, xmax[0]+0.1)
ax.set_ylim(xmin[1]-0.2, xmax[1]+0.1)
plt.tight_layout()
plt.savefig('rmpc.png', dpi=600, bbox_inches='tight')
plt.show()

print(f"  Avg solve time   : {np.mean(solve_times):.4f} s")







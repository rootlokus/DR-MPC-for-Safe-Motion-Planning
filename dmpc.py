# Deterministic MPC

# State   : x = [x, y, vx, vy]
# Input   : u = [ax, ay]
# Dynamics: x_{k+1} = A x_k + B u_k          (no disturbance in prediction)
#           Actual d_k ~ Uniform(-d_bound, d_bound)

# State constraints (hard):
#   x_min <= x_k <= x_max   


import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from scipy.linalg import solve_discrete_are
import time


nx, nu = 4, 2
T_sim  = 70
K_hor  = 8
dt     = 0.1

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
xmin = np.array([-0.1, -0.1])
xmax = np.array([ 3.5,  3.0])

d_bound = 0.1   
dmin    = 0.1    

obs_centers  = [[1.5, 0.5], [1.5, 2.0]]
radii        = [0.3, 0.3]
poly_sides_n = [6, 6]
num_obs      = len(obs_centers)

Q      = np.diag([2000.0, 2000.0, 800.0, 800.0])
R      = 0.5  * np.eye(nu)
S_cost = 50.0 * np.eye(nu)
P_term = solve_discrete_are(A, B, Q, R)

# Obstacle constaints
E_obs, e_obs_nom = [], []
for n in range(num_obs):
    angle   = np.linspace(0, 2*np.pi, poly_sides_n[n]+1)[:-1]
    normals = np.column_stack((np.cos(angle), np.sin(angle)))
    oc      = np.array(obs_centers[n])
    E_obs.append(normals)
    e_obs_nom.append(np.array([
        normals[m] @ (oc + radii[n]*normals[m]) for m in range(len(normals))
    ]))



def build_dmpc(x0_val, x_targ, u_prev):
    opti = ca.Opti()

    X = opti.variable(nx, K_hor+1)   # state trajectory
    U = opti.variable(nu, K_hor)     # inputs

    x0_p    = opti.parameter(nx)
    xref_p  = opti.parameter(nx)
    uprev_p = opti.parameter(nu)

    # Cost
    cost = ca.mtimes([(X[:,K_hor]-xref_p).T, P_term, (X[:,K_hor]-xref_p)])
    for k in range(K_hor):
        cost += ca.mtimes([(X[:,k]-xref_p).T, Q,      (X[:,k]-xref_p)])
        cost += ca.mtimes([U[:,k].T,           ca.DM(R), U[:,k]])
        du    = U[:,k] - (uprev_p if k == 0 else U[:,k-1])
        cost += ca.mtimes([du.T, ca.DM(S_cost), du])
    opti.minimize(cost)

    # Nominal dynamics 
    opti.subject_to(X[:,0] == x0_p)
    for k in range(K_hor):
        opti.subject_to(X[:,k+1] == ca.DM(A) @ X[:,k] + ca.DM(B) @ U[:,k])

    for k in range(K_hor+1):
        opti.subject_to(X[0,k] >= xmin[0])
        opti.subject_to(X[0,k] <= xmax[0])
        opti.subject_to(X[1,k] >= xmin[1])
        opti.subject_to(X[1,k] <= xmax[1])
        if k < K_hor:
            opti.subject_to(U[:,k] >= umin)
            opti.subject_to(U[:,k] <= umax)

    #  Obstacle avoidance 
    #  Direction from obstacle center to current robot position — fixed per
    #  solve, making the constraint linear (convex) in X.

    #  n_lin · X_pos_k  >=  n_lin · oc  +  r  +  dmin
    for obs_idx in range(num_obs):
        oc     = np.array(obs_centers[obs_idx])
        r      = radii[obs_idx]
        diff0  = x0_val[:2] - oc
        dist0  = np.linalg.norm(diff0)

        if dist0 < 1e-6:          # robot exactly at obstacle center 
            continue

        n_lin  = diff0 / dist0    # unit normal pointing away from obstacle
        rhs    = float(n_lin @ oc) + r + dmin   

        for k in range(1, K_hor+1):
            face = n_lin[0]*X[0,k] + n_lin[1]*X[1,k]
            opti.subject_to(face >= rhs)

    opti.set_value(x0_p,    x0_val)
    opti.set_value(xref_p,  x_targ)
    opti.set_value(uprev_p, u_prev)
    opti.set_initial(X, ca.repmat(ca.DM(x0_val), 1, K_hor+1))
    opti.set_initial(U, 0)

    p_opts = {'expand': True, 'print_time': 0}
    s_opts = {
        'max_iter'       : 1000,
        'print_level'    : 0,
        'sb'             : 'yes',
        'tol'            : 1e-4,
        'acceptable_tol' : 1e-3,
        'constr_viol_tol': 1e-4,
    }
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
        return sol.value(U[:,0]), sol.value(X), sol.value(cost), True
    except Exception as ex:
        print(f"  [DMPC] Solver failed: {ex}")
        return np.zeros(nu), None, 0, False


x_real  = np.array([0.0, 0.0, 0.0, 0.0])
x_targ  = np.array([3.0, 2.0, 0.0, 0.0])
u_last  = np.zeros(nu)

hist_x      = [x_real.copy()]
hist_u      = []
sol_x_pred  = []
solve_times = []
costs      =  []

collision_flag = False
failed_flag    = False


for t in range(T_sim):

    # Collision check
    for obs_idx in range(num_obs):
        if np.all(E_obs[obs_idx] @ x_real[:2] <= e_obs_nom[obs_idx]):
            print(f"  t={t}: COLLISION with obstacle {obs_idx}!")
            collision_flag = True
            break
    if collision_flag:
        break

    dist = np.linalg.norm(x_real[:2] - x_targ[:2])
    if dist <= 0.1:
        print(f"  t={t}: GOAL REACHED  dist={dist:.4f}")
        break

    t0 = time.time()
    u_opt, X_pred, cost, ok = build_dmpc(x_real, x_targ, u_last)
    solve_times.append(time.time() - t0)
    costs.append(cost)

    if not ok:
        failed_flag = True
        u_opt = np.zeros(nu)

    sol_x_pred.append(X_pred)

    # Real plant: nominal dynamics + disturbance
    d_real = np.random.uniform(-d_bound, d_bound, nx)
    x_real = A @ x_real + B @ u_opt + d_real
    u_last = u_opt

    hist_x.append(x_real.copy())
    hist_u.append(u_opt.copy())
    print(f"  t={t:3d} | pos=({x_real[0]:.3f},{x_real[1]:.3f}) | "
          f"u=({u_opt[0]:.3f},{u_opt[1]:.3f}) | dist_goal={dist:.3f} | "
          f"solve={solve_times[-1]:.3f}s")

hist_x = np.array(hist_x).T
hist_u = np.array(hist_u).T if hist_u else np.zeros((nu, 1))



#  Plotting results

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
ax.plot(hist_x[0], hist_x[1], 'b-', linewidth=2.5, label='DMPC Trajectory')
ax.plot(hist_x[0,0], hist_x[1,0], 'go', ms=10, label='Start')
ax.plot(x_targ[0],  x_targ[1],   'g*', ms=15, label='Target')

# Predictions
for i, pred in enumerate(sol_x_pred):
    if pred is not None:
        ax.plot(pred[0], pred[1], 'c--', alpha=0.4,
                label='MPC Prediction' if i == 0 else None)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title(f'Deterministic MPC (T={T_sim})')
ax.legend(loc='best')
ax.grid(True)
ax.set_xlim(xmin[0]-0.2, xmax[0]+0.1)
ax.set_ylim(xmin[1]-0.2, xmax[1]+0.1)
plt.tight_layout()
plt.show()

print(f"  Avg solve time   : {np.mean(solve_times):.4f} s")


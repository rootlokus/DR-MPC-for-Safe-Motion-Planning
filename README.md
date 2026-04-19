# DR-MPC for Safe Motion Planning

This repository contains the implementation of various Model Predictive Control (MPC) strategies for safe motion planning, ranging from deterministic baselines to Distributionally Robust (DR) approaches.

## 1. Deterministic MPC
A standard MPC implementation that assumes perfect model knowledge and a static, known environment.

* **Source Code:** [dmpc.py](dmpc.py)
* **Resulting Plot:**
    ![Deterministic MPC Result](Results/dmpc.png)

---

## 2. Robust MPC
An MPC formulation designed to handle bounded uncertainties by optimizing for the worst-case scenario within a set of possible disturbances.

* **Source Code:** [rmpc.py](rmpc.py)
* **Resulting Plot:**
    ![Robust MPC Result](Results/rmpc.png)

---

## 3. DR-MPC without ADF
Distributionally Robust MPC (DR-MPC) implemented without an Adaptive Density Filter. This method optimizes performance across a set of probability distributions to ensure safety under distributional uncertainty.

* **Source Code:** [dr_mpc_no_adf.py](drmpc_comb_no_adf.py)
* **Resulting Plots (Trajectories, Control, and Velocities):**
    ![DR-MPC without ADF Results](Results/SADF_1437_190426_Traj_C.png)
    ![DR-MPC without ADF Results](Results/SADF_1437_190426__Controls_C.png)
    ![DR-MPC without ADF Results](Results/C_SADF_1456_190426__Velocities_C.png)

---

## 4. DR-MPC with ADF
The Distributionally Robust MPC implementation utilizing an Adaptive Density Filter (ADF) for improved state estimation and safety in complex environments.

* **Source Code:** [drmpc_adf.py](drmpc_comb_adf.py)
* **Resulting Plots (Trajectories, Control, and Velocities):**
    ![DR-MPC with ADF Results](Results/C_SADF_1456_190426_Traj_C.png)
    ![DR-MPC with ADF Results](Results/C_SADF_1456_190426__Controls_C.png)
    ![DR-MPC with ADF Results](Results/C_SADF_1456_190426__Velocities_C.png)

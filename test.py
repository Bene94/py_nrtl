import os
import ctypes
import numpy as np
import time
import torch

from calc_lle import calc_LLE


import py_nrtl

def dummy():
    pass


alpha = np.array([
    [0.0, -1.39],
    [-1.39, 0.0]
])

tau = np.array([
    [0.0, 1.2],
    [1.36, 0.0]
])

alpha = np.array([
    [0.0, -1.23],
    [-1.23, 0.0]
])

tau = np.array([
    [0.0, 0.244],
    [1.337, 0.0]
])

alpha = np.array([
    [0.0, 0.1425],
    [0.1425, 0.0]
])

tau = np.array([
    [0.0000, 1.3079],
    [1.3409, 0.0000]
])

alpha_matrix = np.array([[ 
    [0.0, 0.1691],
    [0.1691, 0.0]],
    [[0.0, 0.1691],
    [0.1691, 0.0]]

])

tau_matrix = np.array([[ 
    [0.0, 1.2648],
    [1.3111, 0.0]],
    [[0.0, 1.2648],
    [1.3111, 0.0]]

])

alpha_single = [0.2023, 0.2023, 0.2023]
tau_12 = [1.0023, 1.0023, 1.0023]
tau_21 = [0.9410, 0.9410, 0.9410]


for i in range(alpha.shape[0]): 
    alpha_matrix_i = torch.tensor([[[0, alpha_single[i]],[alpha_single[i], 0]]])
    tau_matrix_i = torch.tensor([[[0, tau_12[i]],[tau_21[i], 0]]])
    if i == 0:
        alpha_matrix = alpha_matrix_i
        tau_matrix = tau_matrix_i
    else:
        alpha_matrix = torch.cat((alpha_matrix, alpha_matrix_i))
        tau_matrix = torch.cat((tau_matrix,tau_matrix_i))

alpha_matrix = alpha_matrix.to(torch.float64)
tau_matrix = tau_matrix.to(torch.float64)

alpha_matrix = torch.tensor([
        [[0.0000, 0.1425],
         [0.1425, 0.0000]],
        [[0.0000, 0.1418],
         [0.1418, 0.0000]]
        ], dtype=torch.float64)

alpha_matrix2 = np.array([
        [[0.0000, 0.1425],
         [0.1425, 0.0000]],
        [[0.0000, 0.1418],
         [0.1418, 0.0000]]])

tau_matrix = torch.tensor([
        [[0.0000, 1.3079],
         [1.3409, 0.0000]],

        [[0.0000, 0.1418],
         [0.1418, 0.0000]]], dtype=torch.float64)

tau_matrix2 = np.array([
        [[0.0000, 1.3079],
         [1.3409, 0.0000]],

        [[0.0000, 0.1418],
         [0.1418, 0.0000]]])


z = np.array([0.5, 0.5])  # Mole fractions for the mixture
z = np.array([0.5111, 0.4889])  # Mole fractions for the mixture
x0 = np.array([0.0, 1])  # Initial guess for phase 1 mole fractions

# Define your parameters here
n = 2
alphas = np.stack([alpha for _ in range(n)], axis=0)
taus = np.stack([tau for _ in range(n)], axis=0)
zs = np.stack([z for _ in range(n)], axis=0)
#z_tensor = torch.tensor(zs)
x0s = np.stack([x0 for _ in range(n)], axis=0)
#x0_tensor = torch.tensor(x0s)

print(py_nrtl.calc_lle_py(alpha, tau, z, x0, 1e-6, 100))
print(calc_LLE(alpha, tau, z, x0))
#out = py_nrtl.calc_lle_par_py(alphas, taus, zs, x0s, 1e-6, 100)
out = py_nrtl.calc_lle_par_py(alpha_matrix.detach().cpu().numpy(), tau_matrix.detach().cpu().numpy(), zs, x0s, 1e-6, 100) # 
print((out[0][0], out[1][0], out[2][0]))
out2 = py_nrtl.calc_lle_par_py(alpha_matrix2, tau_matrix2, zs, x0s, 1e-6, 100)
print((out2[0][0], out2[1][0], out2[2][0]))
out3 = py_nrtl.calc_lle_par_py(alphas, taus, zs, x0s, 1e-6, 100)
print((out3[0][0], out3[1][0], out3[2][0]))
time_start_rust = time.time()
if False:
    for i in range(1000):
        x, y, beta = py_nrtl.calc_lle_py(alpha, tau, z, x0)
    print((x, y, beta))
    time_end_rust = time.time()

    time_start_python = time.time()
    for i in range(1000):
        x, y, beta = calc_LLE(alpha, tau, z, x0)
    print((x, y, beta))
    time_end_python = time.time()

    time_start_empty_loop = time.time()
    a = 0
    for i in range(1000):
        dummy()
    print(a)
    time_end_empty_loop = time.time()


    time_start_rust_par = time.time()
    x_par, y_par, beta_par = py_nrtl.calc_lle_par_py(alphas, taus, zs, x0s, 1e-6, 100)
    time_end_rust_par = time.time()


    print("Runtimes:")
    print("Rust: ", np.round(time_end_rust - time_start_rust,3), "s")
    print("Python: ", np.round(time_end_python - time_start_python,2), "s")
    print("Empty Loop: ", np.round(time_end_empty_loop - time_start_empty_loop,2), "s")
    print("Speedup: ", np.round((time_end_python - time_start_python)/(time_end_rust - time_start_rust),2))


    print("Parallel Runtimes:")
    print("Rust parallel: ", np.round(time_end_rust_par - time_start_rust_par,3), "s")
    print("Speedup: ", np.round((time_end_python - time_start_python)/(time_end_rust_par - time_start_rust_par),2))


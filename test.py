import os
import ctypes
import numpy as np
import time

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
    [0.0, -1.38],
    [-1.38, 0.0]
])

tau = np.array([
    [0.0, 0.72],
    [0.91, 0.0]
])

#alpha = np.array([[0. , 0.23449744],
#                 [0.23449744, 0. ]])
#tau = np.array([[0. , 1.1661877 ],
#                [1.17877817, 0. ]])

#alpha = np.array([[0. , 0.2],
#                 [0.2, 0. ]])
#tau = np.array([[0. , 0.3 ],
#                [0.3, 0. ]])

z = np.array([0.5, 0.5])  # Mole fractions for the mixture
x0 = np.array([0.0, 1])  # Initial guess for phase 1 mole fractions

# Define your parameters here
alphas = np.stack([alpha for _ in range(1000)], axis=0)
taus = np.stack([tau for _ in range(1000)], axis=0)
zs = np.stack([z for _ in range(1000)], axis=0)
x0s = np.stack([x0 for _ in range(1000)], axis=0)

print(py_nrtl.calc_lle_py(alpha, tau, z, x0))
print(calc_LLE(alpha, tau, z, x0))
out = py_nrtl.calc_lle_par_py(alphas, taus, zs, x0s)
print((out[0][0], out[1][0], out[2][0]))
time_start_rust = time.time()
for i in range(1000):
    x, y, __ = py_nrtl.calc_lle_py(alpha, tau, z, x0)
time_end_rust = time.time()

time_start_python = time.time()
for i in range(1000):
    x, y, beta = calc_LLE(alpha, tau, z, x0)
time_end_python = time.time()

time_start_empty_loop = time.time()
a = 0
for i in range(1000):
    dummy()
print(a)
time_end_empty_loop = time.time()


time_start_rust_par = time.time()
x_par, y_par, beta_par = py_nrtl.calc_lle_par_py(alphas, taus, zs, x0s)
time_end_rust_par = time.time()


print("Runtimes:")
print("Rust: ", np.round(time_end_rust - time_start_rust,3), "s")
print("Python: ", np.round(time_end_python - time_start_python,2), "s")
print("Empty Loop: ", np.round(time_end_empty_loop - time_start_empty_loop,2), "s")
print("Speedup: ", np.round((time_end_python - time_start_python)/(time_end_rust - time_start_rust),2))


print("Parallel Runtimes:")
print("Rust parallel: ", np.round(time_end_rust_par - time_start_rust_par,3), "s")
print("Speedup: ", np.round((time_end_python - time_start_python)/(time_end_rust_par - time_start_rust_par),2))

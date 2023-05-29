import os
import ctypes
import numpy as np
import time

from calc_lle import calc_LLE


import py_nrtl


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

alpha = np.array([[0. , 0.23449744],
                 [0.23449744, 0. ]])
tau = np.array([[0. , 1.1661877 ],
                [1.17877817, 0. ]])

alpha = np.array([[0. , 0.2],
                 [0.2, 0. ]])
tau = np.array([[0. , 0.3 ],
                [0.3, 0. ]])

z = np.array([0.5, 0.5])  # Mole fractions for the mixture
x0 = np.array([0.0, 1])  # Initial guess for phase 1 mole fractions

print(py_nrtl.calc_lle_py(alpha, tau, z, x0))
print(calc_LLE(alpha, tau, z, x0))

time_start_rust = time.time()
for i in range(1000):
    x, y, __ = py_nrtl.calc_lle_py(alpha, tau, z, x0)
time_end_rust = time.time()

time_start_python = time.time()
for i in range(1000):
    x, y, beta = calc_LLE(alpha, tau, z, x0)
time_end_python = time.time()

time_start_empty_loop = time.time()
for i in range(1000):
    a = 1+1
time_end_empty_loop = time.time()

print("Runtimes:")
print("Rust: ", np.round(time_end_rust - time_start_rust,3), "ms")
print("Python: ", np.round(time_end_python - time_start_python,2), "ms")
print("Speedup: ", np.round((time_end_python - time_start_python)/(time_end_rust - time_start_rust),2))
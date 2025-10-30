# Evelyn Wilson
# Numerical Methods
# Lab 5 Question 3
# Due October 30, 2025

import math
import numpy as np

# Solve an initial-value problem

# interval 1 <=t <= 2

y1 = np.exp(-1)

def dy_dt(y, t):
    return 2*y*(1/t - t)

def y(t):
    return (t**2)*(np.exp(-t**2))

def euler(n, h, t):
    for j in (1, t:

    return None

def mod_euler(n, h, t):
    return None

def ab_am_2step_pc(n, h, t):
    return None

def rk4(n, h, t):
    return None

for i in range(3, 100):
    h = 2**(-i)
    # Create an array points t from 1 to 2 with delta_t
    t = np.arange(1, 2 + h, h)
    # Call each method to solve the ODE
    euler_solve = euler(i, h, t)
    mod_euler_solve = mod_euler(i, h, t)
    ab_am_solve = ab_am_2step_pc(i, h, t)
    rk4_solve = rk4(i, h, t)




# Plot the solutions vs exact solution

# Plot error for each as a function of t

# Plot the absolute val of error at y(2) vs 1/delta_t w/ log-log scale


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

def euler(n):
    return None

def mod_euler(n):
    return None

def ab_am_2step_pc(n):
    return None

def rk4(n):
    return None

for i in range(3, 100):
    delta_t = 2**(-i)
    

# Plot the solutions vs exact solution

# Plot error for each as a function of t

# Plot the absolute val of error at y(2) vs 1/delta_t w/ log-log scale


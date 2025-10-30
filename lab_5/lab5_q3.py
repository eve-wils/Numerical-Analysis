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

def euler(h, t):
    y_arr = np.zeros(len(t))
    y_arr[0] = y1
    y = y1
        
    for j in range(len(t)-1):
        y = y + h * dy_dt(y, t[j])
        y_arr[j+1] = y
    
    return y_arr

def mod_euler(h, t):
    y_arr = np.zeros(len(t))
    y_arr[0] = y1
    y = y1
    for j in range(len(t)-1):
        y = y + (h/2) * (dy_dt(y, t[j])+ dy_dt(y + h * dy_dt(y, t[j]), t[j+1]))
        y_arr[j+1] = y
    
    return y_arr

# Adams-Bashforth Adams-Moulton 2-step Predictor-Corrector method with a single correction
def ab_am_2step_pc(h, t):
    y_arr = np.zeros(len(t))
    y_arr[0] = y1
    y = y1
    for j in range(len(t)-1):
        if j == 0:
            # Use Euler's method for the first step
            y = y + h * dy_dt(y, t[j])
        else:
            # Predictor step (Adams-Bashforth)
            y_pred = y + (h/2) * (3 * dy_dt(y, t[j]) - dy_dt(y_arr[j-1], t[j-1]))
            # Corrector step (Adams-Moulton)
            y = y + (h/2) * (dy_dt(y, t[j]) + dy_dt(y_pred, t[j+1]))
        y_arr[j+1] = y
    return y_arr

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
    # Compute the exact solution at each point in t
    exact_solution = y(t)
    # Compute the absolute value of the error at each point in t for each method
    euler_error = np.abs(euler_solve - exact_solution)
    mod_euler_error = np.abs(mod_euler_solve - exact_solution)
    ab_am_error = np.abs(ab_am_solve - exact_solution)
    rk4_error = np.abs(rk4_solve - exact_solution)

# Plot the solutions vs exact solution

# Plot error for each as a function of t

# Plot the absolute val of error at y(2) vs 1/delta_t w/ log-log scale


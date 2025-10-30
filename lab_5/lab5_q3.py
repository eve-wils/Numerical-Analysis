# Evelyn Wilson
# Numerical Methods
# Lab 5 Question 3
# Due October 30, 2025

import math
import numpy as np
import matplotlib.pyplot as plt

y1 = np.exp(-1)
max_n = 10

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

def rk4(h, t):
    y_arr = np.zeros(len(t))
    y_arr[0] = y1
    y = y1
    for j in range(len(t)-1):
        t_j = t[j]
        t_mid = t_j + h/2
        t_next = t_j + h
        k1 = dy_dt(y, t_j)
        k2 = dy_dt(y + (h/2) * k1, t_mid)
        k3 = dy_dt(y + (h/2) * k2, t_mid)
        k4 = dy_dt(y + h * k3, t_next)
        y = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        y_arr[j+1] = y

    return y_arr

# Create a new figure for comparing Euler methods with different n values
# Plot exact solution
t_pts = np.linspace(1, 2, 1000)  # Fine grid for smooth exact solution
plt.plot(t_pts, y(t_pts), label='Exact Solution', color='black', linewidth=3)
for i in range(3, max_n):
    h = 2**(-i)
    t = np.arange(1, 2 + h, h)
    euler_sol = euler(h, t)
    plt.plot(t, euler_sol, label=f'n={i}')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Euler's Method vs Exact Solution")
plt.legend()
plt.grid()
plt.show()

# Plot modified Euler solutions for different n values
plt.plot(t_pts, y(t_pts), label='Exact Solution', color='black', linewidth=3)
for i in range(3, max_n):
    h = 2**(-i)
    t = np.arange(1, 2 + h, h)
    mod_euler_sol = mod_euler(h, t)
    plt.plot(t, mod_euler_sol, label=f'n={i}')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Modified Euler's Method vs Exact Solution")
plt.legend()
plt.grid()
plt.show()

# Plot Adams-Bashforth Adams-Moulton 2-step PC solutions for different n values
plt.plot(t_pts, y(t_pts), label='Exact Solution', color='black', linewidth=3)
for i in range(3, max_n):
    h = 2**(-i)
    t = np.arange(1, 2 + h, h)
    ab_am_sol = ab_am_2step_pc(h, t)
    plt.plot(t, ab_am_sol, label=f'n={i}')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("A-B/A-M 2-step PC Method vs Exact Solution")
plt.legend()
plt.grid()
plt.show()

# Plot RK4 solutions for different n values
plt.plot(t_pts, y(t_pts), label='Exact Solution', color='black', linewidth=3)
for i in range(3, max_n):
    h = 2**(-i)
    t = np.arange(1, 2 + h, h)
    rk4_sol = rk4(h, t)
    plt.plot(t, rk4_sol, label=f'n={i}')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("RK4 Method vs Exact Solution")
plt.legend()
plt.grid()
plt.show()

###  Plot error for each method as a function of t ###

# Plot error vs t for each method, showing all n values on the same plot
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

# Euler Method Errors
plt.figure(figsize=(10, 6))
for i, n in enumerate(range(3, max_n)):
    h = 2**(-n)
    t = np.arange(1, 2 + h, h)
    exact = y(t)
    euler_sol = euler(h, t)
    euler_error = np.abs(euler_sol - exact)
    plt.plot(t, euler_error, color=colors[i], label=f'n={n}', alpha=0.7)
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.title("Euler's Method Error vs t for Different n")
plt.legend()
plt.grid(True)
plt.yscale('log')  # Using log scale for better visibility of errors
plt.show()

# Modified Euler Method Errors
plt.figure(figsize=(10, 6))
for i, n in enumerate(range(3, max_n)):
    h = 2**(-n)
    t = np.arange(1, 2 + h, h)
    exact = y(t)
    mod_euler_sol = mod_euler(h, t)
    mod_euler_error = np.abs(mod_euler_sol - exact)
    plt.plot(t, mod_euler_error, color=colors[i], label=f'n={n}', alpha=0.7)
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.title("Modified Euler's Method Error vs t for Different n")
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

# AB-AM Method Errors
plt.figure(figsize=(10, 6))
for i, n in enumerate(range(3, max_n)):
    h = 2**(-n)
    t = np.arange(1, 2 + h, h)
    exact = y(t)
    ab_am_sol = ab_am_2step_pc(h, t)
    ab_am_error = np.abs(ab_am_sol - exact)
    plt.plot(t, ab_am_error, color=colors[i], label=f'n={n}', alpha=0.7)
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.title("A-B/A-M 2-step PC Method Error vs t for Different n")
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

# RK4 Method Errors
plt.figure(figsize=(10, 6))
for i, n in enumerate(range(3, max_n)):
    h = 2**(-n)
    t = np.arange(1, 2 + h, h)
    exact = y(t)
    rk4_sol = rk4(h, t)
    rk4_error = np.abs(rk4_sol - exact)
    plt.plot(t, rk4_error, color=colors[i], label=f'n={n}', alpha=0.7)
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.title("RK4 Method Error vs t for Different n")
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

# Calculate errors at t=2 for convergence analysis
n_values = range(3, max_n)
h_values = [2**(-n) for n in n_values]
euler_errors_at_2 = []
mod_euler_errors_at_2 = []
ab_am_errors_at_2 = []
rk4_errors_at_2 = []

# Store the final error (at t=2) for each method and n value
for n in n_values:
    h = 2**(-n)
    t = np.arange(1, 2 + h, h)
    exact = y(t)
    
    euler_errors_at_2.append(np.abs(euler(h, t)[-1] - exact[-1]))
    mod_euler_errors_at_2.append(np.abs(mod_euler(h, t)[-1] - exact[-1]))
    ab_am_errors_at_2.append(np.abs(ab_am_2step_pc(h, t)[-1] - exact[-1]))
    rk4_errors_at_2.append(np.abs(rk4(h, t)[-1] - exact[-1]))

# Create convergence plot
plt.figure(figsize=(10, 6))
h_reciprocal = [1/h for h in h_values]
plt.loglog(h_reciprocal, euler_errors_at_2, 'o-', label='Euler Method')
plt.loglog(h_reciprocal, mod_euler_errors_at_2, 'o-', label='Modified Euler Method')
plt.loglog(h_reciprocal, ab_am_errors_at_2, 'o-', label='A-B/A-M 2-step PC Method')
plt.loglog(h_reciprocal, rk4_errors_at_2, 'o-', label='RK4 Method')
plt.xlabel('1/h')
plt.ylabel('Absolute Error at t=2')
plt.title('Convergence Analysis (log-log scale)')
plt.legend()
plt.grid(True)
plt.show()
# Lab 4 Question 2
# Evelyn Wilson
# October 15, 2025

import numpy as np
import math
import matplotlib.pyplot as plt

# Link lengths
r1 = 7.10
r2 = 2.36
r3 = 6.68
r4 = 1.94

# Results storage
results = []

# Set initial guesses ONCE - will be carried over
theta2 = math.radians(45)
theta3 = math.radians(180)

def f1(th2, th3):
    return (r2 * np.cos(th2)) + (r3*np.cos(th3)) - (r4*np.cos(theta)) - r1

def f2(th2, th3):
    return (r2 * np.sin(th2)) + (r3*np.sin(th3)) - (r4*np.sin(theta))

def df1_dth2(th2, th3):
    return -r2 * np.sin(th2)

def df1_dth3(th2, th3):
    return -r3 * np.sin(th3)

def df2_dth2(th2, th3):
    return r2 * np.cos(th2)

def df2_dth3(th2, th3):
    return r3 * np.cos(th3)

# Iterate over input angles
for input_angle in range(0, 361):
    theta = math.radians(input_angle)
    
    # Newton's method
    for _ in range(100):
        F = np.array([f1(theta2, theta3), f2(theta2, theta3)])
        
        Jac = np.array([
            [df1_dth2(theta2, theta3), df1_dth3(theta2, theta3)],
            [df2_dth2(theta2, theta3), df2_dth3(theta2, theta3)]
        ])
        
        delta = np.linalg.solve(Jac, F)
        
        theta2 -= delta[0]
        theta3 -= delta[1]
        
        if np.linalg.norm(delta) < 1e-10:
            break
    
    results.append((input_angle, math.degrees(theta2), math.degrees(theta3)))

# Separate x, y values for easier plotting
x_angles = [r[0] for r in results]
phi = [r[1] for r in results]

# Compute derivatives
forward_diff = []
centered_diff = []

# Forward difference
for i in range(len(x_angles) - 1):
    forward_diff.append((phi[i+1] - phi[i]) / 1)

# Centered difference
for i in range(1, len(x_angles) - 1):
    centered_diff.append((phi[i+1] - phi[i-1]) / 2)

# Plot 1: φ vs θ
plt.figure()
plt.plot(x_angles, phi)
plt.title('φ vs θ')
plt.xlabel('θ (degrees)')
plt.ylabel('φ (degrees)')
plt.grid(True)
plt.show()

# Plot 2: Forward and Centered Difference
plt.figure()
plt.plot(x_angles[1:-1], centered_diff, label='Centered Difference')
plt.plot(x_angles[:-1], forward_diff, label='Forward Difference', linestyle='dashed')
plt.title('dφ/dθ')
plt.xlabel('θ (degrees)')
plt.ylabel('dφ/dθ')
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Difference between Forward and Centered
diff_plot = []
for i in range(len(centered_diff)):
    diff_plot.append(abs(forward_diff[i+1] - centered_diff[i]))

plt.figure()
plt.semilogy(x_angles[1:-1], diff_plot)
plt.title('|Forward - Centered|')
plt.xlabel('θ (degrees)')
plt.ylabel('|Forward - Centered| (log scale)')
plt.grid(True)
plt.show()

######## PART II ########

# Compute alpha based on phi
alpha = [x + 149 for x in phi]

# Link lengths for second linkage
r1 = 1.23
r2 = 1.26
r3 = 1.82
r4 = 2.35

# Results storage
results = []

# Set initial guesses ONCE - will be carried over
theta2 = math.radians(45)
theta3 = math.radians(180)

# Iterate over input angles
for input_angle in alpha:
    theta = math.radians(input_angle)
    
    # Newton's method
    for _ in range(100):
        F = np.array([f1(theta2, theta3), f2(theta2, theta3)])
        
        Jac = np.array([
            [df1_dth2(theta2, theta3), df1_dth3(theta2, theta3)],
            [df2_dth2(theta2, theta3), df2_dth3(theta2, theta3)]
        ])
        
        delta = np.linalg.solve(Jac, F)
        
        theta2 -= delta[0]
        theta3 -= delta[1]
        
        if np.linalg.norm(delta) < 1e-10:
            break
    
    results.append((input_angle, theta2, theta3))

# Separate values
beta = [r[1] for r in results]

### Find dβ/dθ ###
b_forward_diff = []
b_centered_diff = []

dtheta_rad = math.radians(1)

# Forward difference
for i in range(len(beta) - 1):
    b_forward_diff.append((beta[i+1] - beta[i]) / dtheta_rad)

# Centered difference
for i in range(1, len(beta) - 1):
    b_centered_diff.append((beta[i+1] - beta[i-1]) / (2 * dtheta_rad))

omega = 550 / 60  # rad/s

db_dt_forward = [omega * diff for diff in b_forward_diff]
db_dt_centered = [omega * diff for diff in b_centered_diff]

### Find d²β/dθ² ###
b2_forward_diff = []
b2_centered_diff = []

# Forward difference
for i in range(len(beta) - 2):
    b2_forward_diff.append((beta[i+2] - 2*beta[i+1] + beta[i]) / (dtheta_rad ** 2))

# Centered difference
for i in range(1, len(beta) - 1):
    b2_centered_diff.append((beta[i+1] - 2*beta[i] + beta[i-1]) / (dtheta_rad**2))

db2_dt2_forward = [(omega ** 2) * diff for diff in b2_forward_diff]
db2_dt2_centered = [(omega ** 2) * diff for diff in b2_centered_diff]

# Figure 4: β vs θ
plt.figure()
plt.plot(x_angles, np.degrees(beta))
plt.title('β vs θ')
plt.xlabel('θ (degrees)')
plt.ylabel('β (degrees)')
plt.grid(True)
plt.show()

# Figure 5: dβ/dt (both methods)
plt.figure()
plt.plot(x_angles[1:-1], db_dt_centered, label='Centered')
plt.plot(x_angles[:-1], db_dt_forward, label='Forward', linestyle='dashed')
plt.title('dβ/dt (Angular Velocity)')
plt.xlabel('θ (degrees)')
plt.ylabel('dβ/dt (rad/s)')
plt.legend()
plt.grid(True)
plt.show()

# Figure 6: d²β/dt² (both methods)
plt.figure()
plt.plot(x_angles[1:-1], db2_dt2_centered, label='Centered')
plt.plot(x_angles[:-2], db2_dt2_forward, label='Forward', linestyle='dashed')
plt.title('d²β/dt² (Angular Acceleration)')
plt.xlabel('θ (degrees)')
plt.ylabel('d²β/dt² (rad/s²)')
plt.legend()
plt.grid(True)
plt.show()

# Figure 7: Difference for dβ/dt
diff_vel = []
for i in range(len(db_dt_centered)):
    diff_vel.append(abs(db_dt_forward[i+1] - db_dt_centered[i]))

plt.figure()
plt.semilogy(x_angles[1:-1], diff_vel)
plt.title('|Forward - Centered| for dβ/dt')
plt.xlabel('θ (degrees)')
plt.ylabel('Difference (rad/s, log scale)')
plt.grid(True)
plt.show()

# Figure 8: Difference for d²β/dt²
diff_acc = []
for i in range(len(db2_dt2_centered) - 1):  # Changed from len(db2_dt2_centered)
    diff_acc.append(abs(db2_dt2_forward[i+1] - db2_dt2_centered[i]))

plt.figure()
plt.semilogy(x_angles[2:-1], diff_acc)  # Changed from x_angles[1:-1]
plt.title('|Forward - Centered| for d²β/dt²')
plt.xlabel('θ (degrees)')
plt.ylabel('Difference (rad/s², log scale)')
plt.grid(True)
plt.show()
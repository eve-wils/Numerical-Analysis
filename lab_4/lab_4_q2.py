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

# Iterate over input angles
for input_angle in range(0, 361):
    theta = math.radians(input_angle)
    
    # Reset initial guesses for each input angle
    theta2 = math.radians(45)
    theta3 = math.radians(360)

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
y_theta2 = [r[1] for r in results]

# Compute derivatives
forward_diff = []
centered_diff = []

# Forward difference
for i in range(len(x_angles) - 1):
    forward_diff.append((y_theta2[i+1] - y_theta2[i]) / 1)  # 1 degree step

# Centered difference
for i in range(1, len(x_angles) - 1):
    centered_diff.append((y_theta2[i+1] - y_theta2[i-1]) / 2)  # 2 degree step

# Plot 1: θ2 vs θ
plt.plot(x_angles, y_theta2)
plt.title('θ2 vs Input Angle')
plt.xlabel('Input Angle (degrees)')
plt.ylabel('θ2 (degrees)')

# Plot 2: Forward Difference
plt.plot(x_angles[:-1], forward_diff, label='Forward Difference')
plt.title('Forward Difference of θ2')
plt.xlabel('Input Angle (degrees)')
plt.ylabel('dθ2/dθ (Forward)')

# Plot 3: Centered Difference
plt.plot(x_angles[1:-1], centered_diff, label='Centered Difference')
plt.title('Centered Difference of θ2')
plt.xlabel('Input Angle (degrees)')
plt.ylabel('dθ2/dθ (Centered)')
plt.show()

# Plot 4: Difference between Forward and Centered
# Align the differences for plotting
diff_plot = []
for i in range(1, len(x_angles) - 1):
    diff_plot.append(abs(forward_diff[i] - centered_diff[i-1]))

plt.semilogy(x_angles[1:-1], diff_plot)
plt.title('Difference between Forward and Centered')
plt.xlabel('Input Angle (degrees)')
plt.ylabel('|Forward - Centered| (log scale)')

plt.tight_layout()
plt.show()

######## PART II ########

# 
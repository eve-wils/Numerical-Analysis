# Lab 2
# Numerical Methods
# Evelyn Wilson
# Due Date: October 1, 2025

# Imports
import math
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# inputting the lengths of the bars
r1 = 43
r2 = 23
r3 =33
r4 = 9

# inputting the angles
theta = math.radians(65)
theta2 = math.radians(52)
theta3 = math.radians(360)

# defining the functions given

def f1 (th2, th3):
    th4 = th2 + th3 - theta
    return (r2 * np.cos(th2)) + (r3*np.cos(th3)) + (r4*np.cos(th4)) - r1
def f2 (th2, th3):
        th4 = th2 + th3 - theta
        return (r2 * np.sin(th2)) + (r3*np.sin(th3)) + (r4*np.sin(th4))

# Do the derivation for each function in respect to each angle
def df1_dth2 (th2, th3):
      th4 = th2 + th3 - theta
      return (-r2 * np.sin(th2)-r4*np.sin(th4))
def df1_dth3 (th2, th3):
      th4 = th2 + th3 - theta
      return (-r3*np.sin(th3)-r4*np.sin(th4))
def df2_dth2 (th2, th3):
      th4 = th2 + th3 - theta
      return r2*np.cos(th2) + r4*np.cos(th4)
def df2_dth3 (th2, th3):
      th4 = th2 + th3 - theta
      return r3*np.cos(th3)+r4*np.cos(th4)
error_arr = []
for i in range(100000):
    # Set of equations
    F = np.array([f1(theta2, theta3), f2(theta2, theta3)])

    # Set up the 2x2 Jacobian
    Jac = np.array([[df1_dth2(theta2, theta3), df1_dth3(theta2, theta3)],
                [df2_dth2(theta2, theta3), df2_dth3(theta2, theta3)]])
    # G(x) = x - [J(x)^-1][F(x)] <- This term can be thought of as delta
    # Solve for delta:
    delta = np.linalg.solve(Jac, F)
    y1 = delta[0]
    y2 = delta[1]
    newtheta2= theta2 - y1
    newtheta3 = theta3 - y2
    # I'm not sure this convergence rate formula is correct
    conv_rate = newtheta2 /(theta2**2)
    conv_rate3 = newtheta3/(theta3**2)
    theta2 = newtheta2
    theta3 = newtheta3


print(f"theta2: {math.degrees(theta2)}, theta3: {math.degrees(theta3)}, conv rate: {conv_rate}, conv rate: {conv_rate3}")
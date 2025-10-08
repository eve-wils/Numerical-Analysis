
# Lab 3 Question 3
# Numerical Methods
# Evelyn Wilson
# Due Date: October 8, 2025

# Imports
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import scipy
import pandas as pd

# Import data
x = [2.9, 2.60, 2.0, 1.5, 1.2, 1.3, 1.8, 2.5, 2.9, 2.9, 2.4, 1.8, 1.3, 1.0]
y = [3.5, 4.05, 4.2, 3.9, 3.4, 2.8, 2.40, 2.25, 1.7, 0.9, 0.55, 0.5, 0.7, 1.2]
data = np.array(list(zip(x, y)))

# define sen(x) - 1
n = len(x) - 1
s = np.linspace(0, 10, len(x))

# Create points at which to evaluate the function
num_points = 500
s_points = np.linspace(0, 10, num_points)

# Define an array to store the calculated points in
x_results = np.zeros(len(s_points))
y_results = np.zeros(len(s_points))

# Cubic Spline Calculation
def cubic_spline(a, s):
    N = len(a) - 1
    h = np.diff(s)
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)

    # Define corner values of A
    A[0,0] = 1
    A[N, N] = 1

    for i in range(1, N):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            b[i] = (3/h[i])*(a[i+1] - a[i]) - (3/h[i-1])*(a[i] - a[i-1])
    c = scipy.linalg.solve(A, b)
    return c, h

def compute_remaining(a, c, h):
    N = len(h)
    b = np.zeros(N)
    d = np.zeros(N)
    for i in range(N):
        b[i] = (a[i+1] - a[i])/h[i] - h[i]*(2*c[i] + c[i+1])/3
        d[i] = (c[i+1] - c[i]) / (3*h[i])
    return b, d

def spline_to_latex(label, a, b, c, d, precision=5):
    """
    Print LaTeX table for cubic spline coefficients.
    a, b, c, d: arrays of coefficients per interval
    label: name of the variable, e.g., 'x' or 'y'
    """
    n = len(b)
    data = {
        "$j$": np.arange(n),
        "$a_j$": np.round(a[:n], precision),
        "$b_j$": np.round(b, precision),
        "$c_j$": np.round(c[:n], precision),
        "$d_j$": np.round(d, precision)
    }
    df = pd.DataFrame(data)
    print(f"\nLaTeX table for {label}(s):\n")
    print(df.to_latex(index=False, escape=False))
    return None

def evaluate_spline(a, b, c, d, s_nodes, s_eval): # where s_eval is the range of points to evaluate at
    vals = []
    for sp in s_eval:
        # find which interval sp lies in
        for j in range(len(s_nodes) - 1):
            if s_nodes[j] <= sp <= s_nodes[j+1]:
                ds = sp - s_nodes[j]
                val = a[j] + b[j]*ds + c[j]*ds**2 + d[j]*ds**3
                vals.append(val)
                break
        else:
            # if sp is beyond last node, use last interval
            ds = sp - s_nodes[-2]
            val = a[-2] + b[-2]*ds + c[-2]*ds**2 + d[-2]*ds**3
            vals.append(val)
    return np.array(vals)

# function calls
cx, hx = cubic_spline(x, s)
cy, hy = cubic_spline(y, s)
bx, dx = compute_remaining(x, cx, hx)
by, dy = compute_remaining(y, cy, hy)

# print the table in LaTeX
spline_to_latex("x", x, bx, cx, dx)
spline_to_latex("y", y, by, cy, dy)

# evaluate the spline and plot it
x_pts = evaluate_spline(x, bx, cx, dx, s, s_points)
y_pts = evaluate_spline(y, by, cy, dy, s, s_points)

plt.plot(s_points, x_pts, label='x(s)')
plt.plot(s, x, 'ko', label='x data')
plt.legend()
plt.title('Interpolation of x')
plt.show()

plt.plot(s_points, y_pts, label='y(s)')
plt.plot(s, y, 'ko', label='y data')
plt.legend()
plt.title('Interpolation of y')
plt.show()

plt.plot(x_pts, y_pts, label='y(x)')
plt.scatter(x, y, label = 'data points')
plt.legend()
plt.title('Interpolation of y(x)')
plt.show()

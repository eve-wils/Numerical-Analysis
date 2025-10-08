
# Lab 3 Question 3
# Numerical Methods
# Evelyn Wilson
# Due Date: October 8, 2025

# Imports
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import scipy

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

cx, hx = cubic_spline(x, s)
cy, hy = cubic_spline(y, s)

def compute_remaining(a, c, h):
    N = len(h)
    b = np.zeros(N)
    d = np.zeros(N)
    for i in range(N):
        b[i] = (a[i+1] - a[i])/h[i] - h[i]*(2*c[i] + c[i+1])/3
        d[i] = (c[i+1] - c[i]) / (3*h[i])
    return b, d


'''
# Call cubic spline function for each point
for i in range(0, len(s_points)):
    # x_results[i] = neville(s, x, s_points[i], order)
    # y_results[i] = neville(s, y, s_points[i], order)

plt.plot(s_points, x_results, label='x(s)')
plt.plot(s, x, 'ko', label='x data')
plt.legend()
plt.title('Interpolation of x')
plt.show()
plt.plot(s_points, y_results, label='y(s)')
plt.plot(s, y, 'ko', label='y data')
plt.legend()
plt.title('Interpolation of y')
plt.show()

plt.plot(x_results, y_results, label='y(x)')
plt.scatter(x, y, label = 'data points')
plt.legend()
plt.title('Interpolation of y')
plt.show()
'''

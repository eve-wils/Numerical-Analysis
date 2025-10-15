# Lab 3 Question 2
# Numerical Methods
# Evelyn Wilson
# Due Date: October 8, 2025

# Imports
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Import data
x = [2.9, 2.60, 2.0, 1.5, 1.2, 1.3, 1.8, 2.5, 2.9, 2.9, 2.4, 1.8, 1.3, 1.0]
y = [3.5, 4.05, 4.2, 3.9, 3.4, 2.8, 2.40, 2.25, 1.7, 0.9, 0.55, 0.5, 0.7, 1.2]
data = np.array(list(zip(x, y)))
order = len(x) - 1
print(order)

# Define neville function
def neville(x, y, x0, n):
    q = np.zeros((n+1, n+1))
    q[:,0] = y
    for j in range(1, n+1):
        for i in range(j, n+1):
            q[i, j] = ((x0 - x[i-j]) * q[i, j-1] - (x0 - x[i]) * q[i-1, j-1]) / (x[i] - x[i-j])
    return q[n, n]

# Create points at which to evaluate the function
s = np.linspace(0, 10, order + 1)

s_points = np.linspace(0, 10, 1000)

# Define an array to store the calculated points in
x_results = np.zeros(len(s_points))
y_results = np.zeros(len(s_points))

# Call neville function for each point
for i in range(0, len(s_points)):
    x_results[i] = neville(s, x, s_points[i], order)
    y_results[i] = neville(s, y, s_points[i], order)

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
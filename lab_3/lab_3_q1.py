# Lab 3
# Numerical Methods
# Evelyn Wilson
# Due Date: October 8, 2025

# Imports
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Define f(x)
def f(x):
    return np.sin(6*x)*np.cos(np.sqrt(5)*x) - (x**2)*(np.e**(-x/5))

# Define neville function
def neville(x, y, x0, n):
    q = np.zeros((n+1, n+1))
    q[:,0] = y
    for j in range(1, n+1):
        for i in range(j, n+1):
            q[i, j] = ((x0 - x[i-j]) * q[i, j-1] - (x0 - x[i]) * q[i-1, j-1]) / (x[i] - x[i-j])
    return q[n, n]

# Define highest n value
highest_n = 10

# Create points at which to evaluate the function
x_points = np.linspace(-2, 2, 1000)

# Iterate through values of n
for n in range(5, highest_n + 1):
    # Define an array to store the calculated points in
    results_arr = np.zeros(len(x_points))

    # Create array of evenly distributed points
    x_array = np.linspace(-2, 2, n+1)
    print(x_array)
    y_array = f(x_array)
    print(y_array)
    for i in range(0, len(x_points)):
        results_arr[i] = neville(x_array, y_array, x_points[i], n)

    plt.plot(x_points, results_arr, label=f'n={n}')
plt.plot(x_points, f(x_points), 'k--', label='f(x)')
plt.legend()
plt.title('Neville Interpolation of f(x)')
plt.show()

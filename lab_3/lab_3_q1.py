# Lab 3 Question 1
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
highest_n = 15

# Create points at which to evaluate the function
num_points = 1000
x_points = np.linspace(-2, 2, num_points)
# Define an array to store the calculated points in
results_arr = np.zeros((highest_n - 5 + 1, len(x_points)))

# Iterate through values of n
for n in range(5, highest_n + 1):
    # Create array of evenly distributed points
    x_array = np.linspace(-2, 2, n+1)
    y_array = f(x_array)

    # Call neville function for each point
    for i in range(0, len(x_points)):
        results_arr[n - 5, i] = neville(x_array, y_array, x_points[i], n)

    plt.plot(x_points, results_arr[n-5,:], label=f'n={n}')

plt.plot(x_points, f(x_points), 'k--', label='f(x)')
plt.legend()
plt.title('Neville Interpolation of f(x)')
plt.show()
# Evaluate f(x) at the points
y_points = f(x_points)
plt.plot(x_points, results_arr[highest_n - 5], label='n=15')
plt.plot(x_points, f(x_points), 'k--', label='f(x)')
plt.legend()
plt.title('Neville interpolation using venly spaced points of f(x)')
plt.show()

'''Repeat with Chebyshev points'''
# Define an array to store the calculated points in
ch_results_arr = np.zeros((highest_n - 5 + 1, len(x_points)))
# Define an array to store the calculated points in
for n in range(5, highest_n + 1):
    # Create array of Chebyshev points, scaling by 2 to get in range [-2, 2]
    x_array = np.cos((2*np.arange(n+1)+1)*np.pi/(2*(n+1))) * 2 
    y_array = f(x_array)

    # Call neville function for each point
    for i in range(0, len(x_points)):
        ch_results_arr[n-5, i] = neville(x_array, y_array, x_points[i], n)

    plt.plot(x_points, ch_results_arr[n-5,:], label=f'n={n}')

plt.plot(x_points, f(x_points), 'k--', label='f(x)')
plt.legend()
plt.title('Chebyshev Interpolation of f(x)')
plt.show()

plt.plot(x_points, ch_results_arr[highest_n - 5], label='n=15')
plt.plot(x_points, f(x_points), 'k--', label='f(x)')
plt.legend()
plt.title('Chebyshev Interpolation of f(x)')
plt.show()

# Plot error of evenly spaced points
dx = 4 / num_points
# evenly spaced
cumul_error_arr = np.zeros((highest_n -4))
# chebyshev
ch_cumul_error_arr = np.zeros((highest_n -4))
for n in range(5, highest_n + 1):
    # evenly spaced
    cumul_error = 0
    # chebyshev
    ch_cumul_error = 0
    for i in range(len(x_points)):
        # evenly spaced
        error_slice = abs(results_arr[n-5, i] - y_points[i])
        rectangle = error_slice*dx
        cumul_error += rectangle
        # chebyshev
        ch_error_slice = abs(ch_results_arr[n-5, i] - y_points[i])
        ch_rectangle = ch_error_slice*dx
        ch_cumul_error += ch_rectangle
    # evenly spaced
    cumul_error_arr[n-5] = cumul_error
    # chebyshev
    ch_cumul_error_arr[n-5] = ch_cumul_error

n_values = np.arange(5, highest_n + 1)
width = 0.35
plt.bar(n_values - width/2, cumul_error_arr, width, color = 'r', label='Evenly Spaced Points')
plt.bar(n_values + width/2, ch_cumul_error_arr, width, color = 'b', label='Chebyshev Optimal Points')
plt.xlabel('n')
plt.ylabel('Cumulative error')
plt.legend()
plt.title('Cumulative Error of Neville Interpolation of f(x)')
plt.show()


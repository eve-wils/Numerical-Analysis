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

# Compute maximum error:
max_errors = np.max(np.abs(results_arr - y_points), axis=1)
print("Maximum errors for evenly spaced points (n=5 to 10):", max_errors)
# Print what n value gives the highest maximum error
max_error_n = np.argmax(max_errors) + 5
print(f"n value with highest maximum error: n={max_error_n}, Error={max_errors[max_error_n-5]}")
# Print what n value gives the lowest maximum error
min_error_n = np.argmin(max_errors) + 5
print(f"n value with lowest maximum error: n={min_error_n}, Error={max_errors[min_error_n-5]}")

# Plot error of evenly spaced points
for n in range(5, highest_n + 1):
    plt.plot(x_points, np.abs(results_arr[n-5,:] - y_points), label=f'n={n}')

plt.yscale('log')
plt.legend()
plt.title('Error of Neville Interpolation of f(x)')
plt.show()

'''Repeat with Chebyshev points'''
# Define an array to store the calculated points in
c_results_arr = np.zeros((highest_n - 5 + 1, len(x_points)))
# Define an array to store the calculated points in
for n in range(5, highest_n + 1):
    # Create array of Chebyshev points, scaling by 2 to get in range [-2, 2]
    x_array = np.cos((2*np.arange(n+1)+1)*np.pi/(2*(n+1))) * 2 
    y_array = f(x_array)

    # Call neville function for each point
    for i in range(0, len(x_points)):
        c_results_arr[n-5, i] = neville(x_array, y_array, x_points[i], n)

    plt.plot(x_points, c_results_arr[n-5,:], label=f'n={n}')

plt.plot(x_points, f(x_points), 'k--', label='f(x)')
plt.legend()
plt.title('Chebyshev Interpolation of f(x)')
plt.show()
# Compute maximum error:
max_errors = np.max(np.abs(c_results_arr - y_points), axis=1)
print("Maximum errors for Chebyshev points (n=5 to 10):", max_errors)
# Print what n value gives the highest maximum error
max_error_n = np.argmax(max_errors) + 5
print(f"n value with highest maximum error: n={max_error_n}, Error={max_errors[max_error_n-5]}")
# Print what n value gives the lowest maximum error
min_error_n = np.argmin(max_errors) + 5
print(f"n value with lowest maximum error: n={min_error_n}, Error={max_errors[min_error_n-5]}")

# Plot error of evenly spaced points
for n in range(5, highest_n + 1):
    plt.plot(x_points, np.abs(c_results_arr[n-5,:] - y_points), label=f'n={n}')

plt.yscale('log')
plt.legend()
plt.title('Error of Neville Interpolation of f(x)')
plt.show()

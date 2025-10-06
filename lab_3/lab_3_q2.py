# Lab 3 Question 1
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
num_points = 1000
s = np.linspace(0, 1, order + 1)
print(s)
s_points = np.linspace(-2, 2, num_points)

# Define an array to store the calculated points in
x_results = np.zeros(len(s_points))
y_results = np.zeros(len(s_points))

# Call neville function for each point
for i in range(0, len(s_points)):
    x_results[i] = neville(s, x, s_points[i], order)


'''
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

''''''Repeat with Chebyshev points''''''
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
plt.show()'''
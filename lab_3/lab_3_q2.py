# Lab 3 Question 1
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
order = len(x) - 1

# Create points at which to evaluate the function
num_points = 500
s = np.linspace(0, 10, n)

s_points = np.linspace(0, 10, num_points)

# Define an array to store the calculated points in
x_results = np.zeros(len(s_points))
y_results = np.zeros(len(s_points))

# Cubic Spline Calculation
def cubic_spline(a, n):
    matrix_A = np.zeros((n, n))
    h = np.empty((n-1))
    for j in range(n): # row
        for k in range(n): # column
            if(j == k and (j == 0 or j == n)): # diagonal end-points
                matrix_A[j, k] = 1
            elif(j == k): # rest of diagonal
                matrix_A[j, k] = 2*(h[j-1]+h[j])
            elif(k == j - 1):
                matrix_A[j, k] = h[k]
            elif(k == j + 1):
                matrix_A[j, k] = h[j]

    b = np.zeros(n)
    for i in range(1, n-1):
        b = (3 / h[i])*(a[i+1] - a[i]) - (3 / h[i-1])*(a[i]-a[i-1])

    c = scipy.linalg.solve(matrix_A, b)
    print(c)

cubic_spline(x, n)
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
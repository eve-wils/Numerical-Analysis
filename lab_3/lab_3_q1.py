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

# Create evenly spaced points along the interval and evaluate the function at those points
j = 0
xj = f(j)
xrange = np.linspace(-2, 2, 1000)
highest_n = 10
results_lst = []
error_lst = []
l = 0
m = 0
for n in range(5, highest_n + 1): # Iterate through values of N to experiment with different orders of polynomials
    nrange = np.linspace(-2.0, 2.0, n + 1) # Creates n+1 points along the interval
    n_result = []
    n_error = []
    for x in xrange:
        product = 1 # Resets the initial value for the product for every new x value to the function evaluated at x
        sum = 0
        for j in range(n+1):
            for k in range(n+1): # Doing the product N times for each x value
                if k != j:
                    product *= ((x - nrange[k])) / (nrange[j] - nrange[k])
            sum += f(nrange[j]) * product
        n_result.append(sum)
        n_error.append(abs(sum - f(x)))
    results_lst.append(n_result)
    error_lst.append(n_error)

results_arr = np.array(results_lst)
error_arr = np.array(error_lst)

p = highest_n - 5 + 1
for i in range(p):
    plt.plot(xrange, results_arr[i], label=f"N={i+5}")

plt.plot(xrange, f(xrange), 'k--', label="f(x)")

plt.yscale("log")
plt.show()

for i in range(p):
    plt.plot(xrange, error_arr[i], label=f"N={i+5}")

plt.yscale("log")
plt.show()
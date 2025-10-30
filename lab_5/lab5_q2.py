# Lab 5 Question 2
# Evelyn Wilson
# Due: October 30, 2025
from numpy.polynomial.legendre import leggauss
import numpy as np
def f(x):
    # Use numpy's exp so this works with array inputs
    return x**2 * np.exp(-x)

def g(x):
    return x**(1/3)

# Calculate integral using Gaussian quadrature with n points
def gaussian_integrate(function, a, b, n):
    x, w = leggauss(n)  # x are points, w are weights
    # scale the points to fit interval [a,b]
    y = (a+b)/2 + ((b-a)*x)/2
    scale = (b-a)/2
    if function == "f":
        return scale * np.dot(w, f(y))
    elif function == "g":
        return scale * np.dot(w, g(y))

def iterate_gaussian(function, a, b):
    old = gaussian_integrate(function, a, b, 1)
    for i in range(2, 300):
        current = gaussian_integrate(function, a, b, i)
        if abs(old - current) <= 1e-9:
            return i, current
        old = current  # Update old value for next iteration
    # If no convergence after 100 iterations, return best estimate
    return 300, current

# Test integrals over [0,1] and [1,2]
print("Integration results:")
print(f"f(x) over [0,1]: {iterate_gaussian('f', 0, 1)}")
print(f"f(x) over [1,2]: {iterate_gaussian('f', 1, 2)}")
print(f"g(x) over [0,1]: {iterate_gaussian('g', 0, 1)}")
print(f"g(x) over [1,2]: {iterate_gaussian('g', 1, 2)}")
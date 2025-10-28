# Lab 5 Question 2
# Evelyn Wilson
# Due: October 30, 2025

import math
import numpy as np

def f(x):
    return x**2 * math.e**(-x)

def g(x):
    return x**(1/3)

# Trapezoidal rule
def trap(function, a, b, n):
    h = (b - a)/n
    x = np.linspace(a, b, n+1)  # Changed to linspace to include endpoint
    
    if function == "f":
        y = f(x)
    elif function == "g":
        y = g(x)
    return (h/2)*(y[0]+2*np.sum(y[1:-1])+y[-1])  # Fixed parentheses and multiplication

# Romberg integration
def romberg(function, a, b):
    tolerance = 1e-9
    max = 10
    romberg = np.zeros((max, max))
    for i in range(max):
        n = 2**i
        romberg[i, 0] = trap(function, a, b, n)
        for j in range(1, i+1):  # Changed to i+1 to include current level
            romberg[i, j] = (4**j * romberg[i, j-1] - romberg[i-1, j-1]) / (4**j - 1)
        if i > 0 and abs(romberg[i, i] - romberg[i-1, i-1]) <= tolerance:  # Added i > 0 check
            print(f"Converged on the {i}th iteration.")
            return i, romberg[i, i]
    return None

print(romberg("f", 0, 1))
print(romberg("g", 0, 1))
print(romberg("f", 1, 2))
print(romberg("g", 1, 2))
n = 3
print(trap("f", 0, 1, n))
print(trap("g", 0, 1, n))
print(trap("f", 1, 2, n))
print(trap("g", 1, 2, n))
# Lab 5 Question 2
# Evelyn Wilson
# Due: October 30, 2025

import math

def f(x):
    return x**2 * math.e**(-x)

def g(x):
    return x**(1/3)

# Trapezoidal rule
def trap(function, start, end):
    return None

# Romberg integration
def romberg(function, start, end):
    # Uses the trapezoidal rule function
    return None

def comp_trap(function, start, end):
    return None

print(romberg("f", 0, 1))
print(romberg("g", 0, 1))
print(romberg("f", 1, 2))
print(romberg("g", 1, 2))

print(trap("f", 0, 1))
print(trap("g", 0, 1))
print(trap("f", 1, 2))
print(trap("g", 1, 2))

print(comp_trap("f", 0, 1))
print(comp_trap("g", 0, 1))
print(comp_trap("f", 1, 2))
print(comp_trap("g", 1, 2))
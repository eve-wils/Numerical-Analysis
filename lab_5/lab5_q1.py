# Lab 5 Question 2
# Evelyn Wilson
# Due: October 30, 2025

import math
import numpy as np

def f(x):
    return x**2 * np.exp(-x)

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
    max = 100
    romberg = np.zeros((max, max))
    for i in range(max):
        n = 2**i
        romberg[i, 0] = trap(function, a, b, n)
        for j in range(1, i+1):  # Changed to i+1 to include current level
            romberg[i, j] = (4**j * romberg[i, j-1] - romberg[i-1, j-1]) / (4**j - 1)
        if i > 0 and abs(romberg[i, i] - romberg[i-1, i-1]) <= tolerance:  # Added i > 0 check
            return i, romberg[i, i]
    return None

nf1, answerf1 = romberg("f", 0, 1)
nf2, answerf2 = romberg("f", 1, 2)

trapf1 = trap("f", 0, 1, nf1)
trapf2 = trap("f", 1, 2, nf2)
print(f"PART A: 0 -> 1; Romberg answer: {answerf1}, Iterations: {nf1}, Trap answer: {trapf1}, Difference = {abs(trapf1 - answerf1)}")
print(f"PART A: 1 -> 2; Romberg answer: {answerf2}, Iterations: {nf2}, Trap answer: {trapf2}, Difference = {abs(trapf2 - answerf2)}")

# PART B calculations
ng1, answerg1 = romberg("g", 0, 1)
ng2, answerg2 = romberg("g", 1, 2)

trapg1 = trap("g", 0, 1, ng1)
trapg2 = trap("g", 1, 2, ng2)
print(f"PART B:  0 -> 1; Romberg answer: {answerg1}, Iterations: {ng1}, Trap answer: {trapg1}, Difference = {abs(trapg1 - answerg1)}")
print(f"PART B:  1 -> 2; Romberg answer: {answerg2}, Iterations: {ng2}, Trap answer: {trapg2}, Difference = {abs(trapg2 - answerg2)}")

# Part a 0 to 1: 
def trap_optimization(function, a, b, max, min = 5):
    for i in range(min, max):
        trap_calc = trap(function, a, b, i)
        _, rom_calc = romberg(function, a, b)
        if abs(trap_calc - rom_calc) < 1e-9:
            return i, trap_calc
    print("Not found")
    return None

print(trap_optimization("f", 0, 1, 6000, 5))
print(trap_optimization("f", 1, 2, 6000, 5))
print(trap_optimization("g", 0, 1, 1700, 1500))
print(trap_optimization("g", 1, 2, 1700, 1500))
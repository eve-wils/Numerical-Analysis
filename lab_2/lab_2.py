# Lab 2
# Numerical Methods
# Evelyn Wilson
# Due Date: October 1, 2025

# Imports
import math
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Define given functions 
def fa (x):
    return ((x + math.cos(x)) * (math.exp(-x ** 2))) + (x * math.cos(x))

def fb(x):
    return fa(x) ** 2

def fc(x):
    return fa(x) ** 3

# Define initial value (plotted on desmos)
x0 = 1.63
x1 = 1.64

''' Secant Method '''

def secant_a (pn, p0):
    return pn - ((fa(pn) * (pn - p0)) / (fa(pn) - fa(p0)))
def secant_b (pn, p0):
    return pn - ((fb(pn) * (pn - p0)) / (fb(pn) - fb(p0)))
def secant_c (pn, p0):
    return pn - ((fc(pn) * (pn - p0)) / (fc(pn) - fc(p0)))

''' Newton's Method '''

# Since we require f'(p) now:
def fa_prime (x):
    return (math.exp(-x**2) * (1 - math.sin(x))) - (2 * math.exp(-x**2) * x * (x + math.cos(x))) + math.cos(x) - (x * math.sin(x))

def fb_prime (x):
    part_1 = ((x + math.cos(x)) * math.exp(-x**2)) + (x * math.cos(x))
    part_2 = ((math.exp(-x**2) * (1 - math.sin(x))) - (2 * math.exp(-x**2) * x * (x + math.cos(x)))+ math.cos(x) - x * math.sin(x))
    return 2 * (part_1 * part_2)

def fc_prime (x):
    part_1 = ((x + math.cos(x)) * math.exp(-x**2)) + (x * math.cos(x))
    part_2 = ((math.exp(-x**2) * (1 - math.sin(x))) - (2 * math.exp(-x**2) * x * (x + math.cos(x)))+ math.cos(x) - x * math.sin(x))
    return 3 * ((part_1** 2) * part_2)

'''
def newt_a ():
    return 
def newt_b ():
def newt_c ():

 Modified Newton's Method
def mod_newt_a ():
def mod_newt_b ():
def mod_newt_c ():

# Cubic Newton's Method
def cubic_newt_a ():
def cubic_newt_a ():
def cubic_newt_a ():
'''
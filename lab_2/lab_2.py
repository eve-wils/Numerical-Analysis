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

def fb (x):
    return fa(x) ** 2

def fc (x):
    return fa(x) ** 3

# Define error function to determine how close a solution is
def abs_error(sol, funct_type):
    if (funct_type == 'a'):
        return  abs(fa(sol) - 0)
    elif (funct_type == 'b'):
        return  abs(fb(sol) - 0)
    elif (funct_type == 'c'):
        return  abs(fc(sol) - 0)
    else:
        return "ERROR: incorrect function type"

# Define error threshold
acceptable_error = 1e-15
best_index = 0
# Function to find order of convergence from data:
def estimate_order(error_arr):
    thresh = 1e-16
    length = len(error_arr)
    vals = []
    best_alpha = 0
    for i in range(length-2):
        err0, err1, err2 = error_arr[i], error_arr[i+1], error_arr[i+2]
        if min(err0, err1, err2) <= 1e-12 or max(err0, err1, err2) >= 1e-2:
            continue
        alpha = math.log(err2/err1) / math.log(err1/err0)
        vals.append(alpha)
        diff1 = abs(2 - alpha)
        if (alpha > best_alpha and alpha <= 2):
            best_alpha = alpha
    return best_alpha


# Define initial value (plotted on desmos)
x0 = 1.63
x1 = 1.64
sol = x0

''' Secant Method '''
# Define secant functions
def secant_a (pn, p0):
    if(fa(pn) - fa(p0) != 0):
        return pn - ((fa(pn) * (pn - p0)) / (fa(pn) - fa(p0)))
    else:
        return 0
def secant_b (pn, p0):
    if(fb(pn) - fb(p0) != 0):
        return pn - ((fb(pn) * (pn - p0)) / (fb(pn) - fb(p0)))
    else:
        return 0
def secant_c (pn, p0):
    if(fc(pn) - fc(p0) != 0):
        return pn - ((fc(pn) * (pn - p0)) / (fc(pn) - fc(p0)))
    else:
        return 0

# Run secant a
print("Starting Secant Method A...")
i_a = 0
x0a = 1.63
x1a = 1.64
sol_a = x0a

sec_err_a = [abs_error(sol_a, 'a')]

while(abs_error(sol_a, 'a') > acceptable_error):

    # Alternate between replacing x0 and x1 with the new value, therefore alternating which parameter comes first
    if(i_a % 2 == 0):
        x0a = secant_a(x1a, x0a)
        sol_a = x0a
        # x0 is now p_n, x1 is p_(n-1)
    else:
        x1a = secant_a(x0a, x1a)
        sol_a = x1a
        # x1 is now p_n, x0 is now p_(n-1), the other branch runs next

    # Increment, so we know how many it takes at the end.
    i_a = i_a + 1

    error_a = abs_error(sol_a, 'a')
    sec_err_a.append(error_a)
print(f"Solution: {sol_a}")
print(f"Iterations: {i_a}")
print(f"Absolute Error: {error_a}")
print(f"Estimated order: {estimate_order(sec_err_a)}")

# Run secant b
print("Starting Secant Method B")
i_b = 0
x0b = 1.63
x1b = 1.64
sol_b = x0b
sec_err_b = [abs_error(sol_b, 'b')]

while(abs_error(sol_b, 'b') > acceptable_error):

    # Alternate between replacing x0 and x1 with the new value, therefore alternating which parameter comes first
    if(i_b % 2 == 0):
        x0b = secant_b(x1b, x0b)
        sol_b = x0b
        # x0 is now p_n, x1 is p_(n-1)
    else:
        x1b = secant_b(x0b, x1b)
        sol_b = x1b
        # x1 is now p_n, x0 is now p_(n-1), the other branch runs next

    # Increment, so we know how many it takes at the end.
    i_b = i_b + 1
    error_b = abs_error(sol_b, 'b')
    sec_err_b.append(error_b)
print(f"Solution: {sol_b}")
print(f"Iterations: {i_b}")
print(f"Absolute Error: {error_b}")
print(f"Estimated order: {estimate_order(sec_err_b)}")

# Run secant c
print("Starting Secant Method C")
i_c = 0
x0c = 1.63
x1c = 1.64
sol_c = x0c
sec_err_c = [abs_error(sol_c, 'c')]
while(abs_error(sol_c, 'c') > acceptable_error):
    # Alternate between replacing x0 and x1 with the new value, therefore alternating which parameter comes first
    if(i_c % 2 == 0):
        x0c = secant_c(x1c, x0c)
        sol_c = x0c
        # x0 is now p_n, x1 is p_(n-1)
    else:
        x1c = secant_c(x0c, x1c)
        sol_c = x1c
        # x1 is now p_n, x0 is now p_(n-1), the other branch runs next

    # Increment, so we know how many it takes at the end.
    i_c = i_c + 1
    error_c = abs_error(sol_c, 'c')
    sec_err_c.append(error_c)
print(f"Solution: {sol_c}")
print(f"Iterations: {i_c}")
print(f"Absolute Error: {error_c}")
print(f"Estimated order: {estimate_order(sec_err_c)}")

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

def newt_a (x):
    return x - (fa(x) / fa_prime(x))
def newt_b (x):
    return x - (fb(x) / fb_prime(x))
def newt_c (x):
    return x - (fc(x) / fc_prime(x))

# Run Newton A
print("***Newton Method A***")
i_a = 0
pn= 1.63
newt_err_a = [abs_error(pn, 'a')]
while(abs_error(pn, 'a') > acceptable_error):
    pn = newt_a(pn)
    i_a = i_a + 1
    error_a = abs_error(pn, 'a')
    newt_err_a.append(error_a)
print(f"Solution: {pn}")
print(f"Iterations: {i_a}")
print(f"Absolute Error: {error_a}")
print(f"Estimated order: {estimate_order(newt_err_a)}")
# Run Newton B
print("***Newton Method B***")
i_b = 0
pn= 1.63
newt_err_b = [abs_error(pn, 'b')]
while(abs_error(pn, 'b') > acceptable_error):
    pn = newt_b(pn)
    i_b = i_b + 1
    error_b = abs_error(pn, 'b')
    newt_err_b.append(error_b)
print(f"Solution: {pn}")
print(f"Iterations: {i_b}")
print(f"Absolute Error: {error_b}")
print(f"Estimated order: {estimate_order(newt_err_b)}")
# Run Newton C
print("***Newton Method C***")
i_c = 0
pn= 1.63
newt_err_c = [abs_error(pn, 'c')]
while(abs_error(pn, 'c') > acceptable_error):
    pn = newt_c(pn)
    i_c = i_c + 1
    error_c = abs_error(pn, 'c')
    newt_err_c.append(error_c)
print(f"Solution: {pn}")
print(f"Iterations: {i_c}")
print(f"Absolute Error: {error_c}")
print(f"Estimated order: {estimate_order(newt_err_c)}")


'''Modified Newton's Method'''
def u_a (x):
    return fa(x) / fa_prime(x)
def u_b (x):
    return fa(x) / fa_prime(x)
def u_c (x):
    return fa(x) / fa_prime(x)

def mod_newt_a (x):
    m = 1 # multiplicity is 1 because there is only 1 root
    return x - (m * u_a(x))
def mod_newt_b (x):
    m = 1 # multiplicity is 1 because there is only 1 root
    return x - (m * u_b(x))
def mod_newt_c (x):
    m = 1 # multiplicity is 1 because there is only 1 root
    return x - (m * u_c(x))

# Run Mod Newton A
print("***Mod. Newton Method A***")
i_a = 0
pn= 1.63
mod_newt_err_a = [abs_error(pn, 'a')]
while(abs_error(pn, 'a') > acceptable_error):
    pn = mod_newt_a(pn)
    i_a = i_a + 1
    error_a = abs_error(pn, 'a')
    mod_newt_err_a.append(error_a)
print(f"Solution: {pn}")
print(f"Iterations: {i_a}")
print(f"Absolute Error: {error_a}")
print(f"Estimated order: {estimate_order(mod_newt_err_a)}")

# Run Mod Newton B
print("***Mod Newton Method B***")
i_b = 0
pn= 1.63
mod_newt_err_b = [abs_error(pn, 'b')]
while(abs_error(pn, 'b') > acceptable_error):
    pn = mod_newt_b(pn)
    i_b = i_b + 1
    error_b = abs_error(pn, 'b')
    mod_newt_err_b.append(error_b)
print(f"Solution: {pn}")
print(f"Iterations: {i_b}")
print(f"Absolute Error: {error_b}")
print(f"Estimated order: {estimate_order(mod_newt_err_b)}")

# Run Mod Newton C
print("***Mod Newton Method C***")
i_c = 0
pn= 1.63
mod_newt_err_c = [abs_error(pn, 'c')]
while(abs_error(pn, 'c') > acceptable_error):
    pn = mod_newt_c(pn)
    i_c = i_c + 1
    error_c = abs_error(pn, 'c')
    mod_newt_err_c.append(error_c)
print(f"Solution: {pn}")
print(f"Iterations: {i_c}")
print(f"Absolute Error: {error_c}")
print(f"Estimated order: {estimate_order(mod_newt_err_c)}")

'''Cubic Newton's Method'''

def cubic_newt_a ():
    return None
def cubic_newt_a ():
    return None
def cubic_newt_a ():
    return None

sec_a = np.arange(len(sec_err_a))
newt_a = np.arange(len(newt_err_a))
mod_newt_a = np.arange(len(mod_newt_err_a))
plt.plot(sec_a, sec_err_a, label='Secant A', color='blue')
plt.plot(newt_a, newt_err_a, label='Newton A', color='red')
plt.plot(mod_newt_a, mod_newt_err_a, label='Mod Newton A', color='yellow')
# Make the graph readable
plt.title("Absolute error for four methods on function A")
plt.xlabel("Iterations")
plt.ylabel("Absolute Error")
plt.yscale("log") # Make it logarithmic
plt.legend()
plt.grid(True)
plt.show()

sec_b = np.arange(len(sec_err_b))
newt_b = np.arange(len(newt_err_b))
mod_newt_b = np.arange(len(mod_newt_err_b))
plt.plot(sec_b, sec_err_b, label='Secant B', color='blue')
plt.plot(newt_b, newt_err_b, label='Newton B', color='red')
plt.plot(mod_newt_b, mod_newt_err_b, label='Mod Newton B', color='yellow')
plt.title("Absolute error for four methods on function B")
plt.xlabel("Iterations")
plt.ylabel("Absolute Error")
plt.yscale("log") # Make it logarithmic
plt.legend()
plt.grid(True)
plt.show()

sec_c = np.arange(len(sec_err_c))
newt_c = np.arange(len(newt_err_c))
mod_newt_c = np.arange(len(mod_newt_err_c))
plt.plot(sec_c, sec_err_c, label='Secant C', color='blue')
plt.plot(newt_c, newt_err_c, label='Newton C', color='red')
plt.plot(mod_newt_c, mod_newt_err_c, label='Mod Newton C', color='yellow')
plt.title("Absolute error for four methods on function C")
plt.xlabel("Iterations")
plt.ylabel("Absolute Error")
plt.yscale("log") # Make it logarithmic
plt.legend()
plt.grid(True)
plt.show()

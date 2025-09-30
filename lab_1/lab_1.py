# Lab 1
# Numerical Analysis
# Evelyn Wilson
# Due Date: September 24

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Define Bessel Function
def bessel_function (jn_minus, jn, n, x):
    return (((2 * n) / x) * jn) - jn_minus

# Call bessel function forwards (store into array) for x = 1
bessel_forwards_1 = np.zeros(51)
bessel_forwards_1[0] = 7.6520e-01
bessel_forwards_1[1] = 4.4005e-01

# Iterate through range, calculating Bessel function values and storing into an array
for n in range(2, 51):
    bessel_forwards_1[n] = bessel_function(bessel_forwards_1[n-2], bessel_forwards_1[n-1], n, 1)

# Calculate the absolute error using those calculated values and the ideal values, storing into an array
bessel_forwards_1_abs = np.zeros(51)
for n in range(0, 51):
    bessel_forwards_1_abs[n] = abs(special.jn(n, 1) - bessel_forwards_1[n])

# Calculate the relative error using the same calculated values and the same ideal values(but divided by the ideal value) and store into an array
bessel_forwards_1_rel = np.zeros(51)
for n in range(0, 51):
    if special.jn(n, 1) != 0:
        bessel_forwards_1_rel[n] = abs((special.jn(n, 1) - bessel_forwards_1[n])/ special.jn(n, 1)) 


# Call bessel function forwards (store into array) for x = 15
bessel_forwards_15 = np.zeros(51)
bessel_forwards_15[0] = -1.4224e-02
bessel_forwards_15[1] = 2.0510e-01

# Do the same stuff as above but with x = 15
for n in range(2, 51):
    bessel_forwards_15[n] = bessel_function(bessel_forwards_15[n-2], bessel_forwards_15[n-1], n, 15)

bessel_forwards_15_abs = np.zeros(51)
for n in range(0, 51):
    bessel_forwards_15_abs[n] = abs(special.jn(n, 15) - bessel_forwards_15[n])

bessel_forwards_15_rel = np.zeros(51)
for n in range(0, 51):
    if special.jn(n, 15) != 0:
        bessel_forwards_15_rel[n] = abs((special.jn(n, 15) - bessel_forwards_15[n])/ special.jn(n, 15)) 


# Call bessel function forwards (store into array) for x = 40
bessel_forwards_40 = np.zeros(51)
bessel_forwards_40[0] = 7.3669e-03
bessel_forwards_40[1] = 1.2604e-01

# Same stuff, different number
for n in range(2, 51):
    bessel_forwards_40[n] = bessel_function(bessel_forwards_40[n-2], bessel_forwards_40[n-1], n, 40)

bessel_forwards_40_abs = np.zeros(51)
for n in range(0, 51):
    bessel_forwards_40_abs[n] = abs(special.jn(n, 40) - bessel_forwards_40[n])

bessel_forwards_40_rel = np.zeros(51)
for n in range(0, 51):
    if special.jn(n, 40) != 0:
        bessel_forwards_40_rel[n] = abs((special.jn(n, 40) - bessel_forwards_40[n])/ special.jn(n, 40)) 

# Plot the difference between forwards and ideal
x = np.arange(len(bessel_forwards_1_abs))
plt.plot(x, bessel_forwards_1_abs, label='x=1', color='blue') # MatLab roleplay
plt.plot(x, bessel_forwards_15_abs, label='x=15', color='red')
plt.plot(x, bessel_forwards_40_abs, label='x=40', color='yellow')
# Make the graph readable
plt.title("Absolute error for forward Bessel function")
plt.xlabel("n")
plt.ylabel("Absolute Error")
plt.yscale("log") # Make it logarithmic
plt.legend()
plt.grid(True)
plt.show()

x = np.arange(len(bessel_forwards_1_rel)) # Create an x-axis that imitates the dataset we're graphing
plt.plot(x, bessel_forwards_1_rel, label='x=1', color='blue')
plt.plot(x, bessel_forwards_15_rel, label='x=15', color='red')
plt.plot(x, bessel_forwards_40_rel, label='x=40', color='yellow')
# Make the graph readable
plt.title("Relative error for forward Bessel function")
plt.xlabel("n")
plt.ylabel("Relative Error")
plt.yscale("log") # It's here too!
plt.legend()
plt.grid(True)
plt.show()


### BACKWARDS ###

# Define a backwards Bessel function in which Jn-1 is solved for instead of Jn+1
def bw_bessel_function (jn, jn_plus, n, x):
    return (((2 * n) / x) * jn) - jn_plus

# Call bessel function backwards (store into array) for x = 1
bessel_backwards_1 = np.zeros(51)
bessel_backwards_1[50] = 2.9060e-80
bessel_backwards_1[49] = 2.9057e-78

# Calculate Bessel function value for each n
for n in range(48, -1, -1):
    bessel_backwards_1[n] = bw_bessel_function(bessel_backwards_1[n+1], bessel_backwards_1[n+2], n, 1)

# Calculate absolute error for each n and store it into an array so you can plot it
bessel_backwards_1_abs = np.zeros(51)
for n in range(50, -1, -1):
    bessel_backwards_1_abs[n] = abs(special.jn(n, 1) - bessel_backwards_1[n])

# Calculate relative error for each n and store it into an array for plotting
bessel_backwards_1_rel = np.zeros(51)
for n in range(50, -1, -1):
    if special.jn(n, 1) != 0:
        bessel_backwards_1_rel[n] = abs((special.jn(n, 1) - bessel_backwards_1[n])/ special.jn(n, 1)) 



# Call bessel function backwards (store into array) for x = 15
bessel_backwards_15 = np.zeros(51)
bessel_backwards_15[50] = 6.1061e-22
bessel_backwards_15[49] = 3.9789e-21

# Calculate Bessel function value for each n
for n in range(48, -1, -1):
    bessel_backwards_15[n] = bw_bessel_function(bessel_backwards_15[n+1], bessel_backwards_15[n+2], n, 15)

# Calculate absolute error for each n
bessel_backwards_15_abs = np.zeros(51)
for n in range(50, -1, -1):
    bessel_backwards_15_abs[n] = abs(special.jn(n, 15) - bessel_backwards_15[n])

# Calculate  relative error for each n
bessel_backwards_15_rel = np.zeros(51)
for n in range(50, -1, -1):
    if special.jn(n, 15) != 0:
        bessel_backwards_15_rel[n] = abs((special.jn(n, 15) - bessel_backwards_15[n])/ special.jn(n, 15)) 

# Call bessel function backwards (store into array) for x = 40
bessel_backwards_40 = np.zeros(51)
bessel_backwards_40[50] = 6.8185e-04
bessel_backwards_40[49] = 1.3775e-03

# Calculate value of Bessel function for each n
for n in range(48, -1, -1):
    bessel_backwards_40[n] = bw_bessel_function(bessel_backwards_40[n+1], bessel_backwards_40[n+2], n, 40)

# Calculate absolute error for each n
bessel_backwards_40_abs = np.zeros(51)
for n in range(50, -1, -1):
    bessel_backwards_40_abs[n] = abs(special.jn(n, 40) - bessel_backwards_40[n])

# Calculate relative error for each n
bessel_backwards_40_rel = np.zeros(51)
for n in range(50, -1, -1):
    if special.jn(n, 40) != 0:
        bessel_backwards_40_rel[n] = abs((special.jn(n, 40) - bessel_backwards_40[n])/ special.jn(n, 40)) 

# Plot the absolute error (same way we did the forward one)
x = np.arange(len(bessel_backwards_1_abs))
plt.plot(x, bessel_backwards_1_abs, label='x=1', color='blue')
plt.plot(x, bessel_backwards_15_abs, label='x=15', color='red')
plt.plot(x, bessel_backwards_40_abs, label='x=40', color='yellow')
plt.title("Absolute error for backward Bessel function")
plt.xlabel("n")
plt.ylabel("Absolute Error")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()

# Plot the relative error (same way we did the forward one)
x = np.arange(len(bessel_backwards_1_rel))
plt.plot(x, bessel_backwards_1_rel, label='x=1', color='blue')
plt.plot(x, bessel_backwards_15_rel, label='x=15', color='red')
plt.plot(x, bessel_backwards_40_rel, label='x=40', color='yellow')
plt.title("Relative error for backward Bessel function")
plt.xlabel("n")
plt.ylabel("Relative Error")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()


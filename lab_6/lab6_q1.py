# Evelyn Wilson
# Numerical Methods
# Lab 6 Question 1
# Due November 6

import numpy as np

# Function definitions

def Rk4(alpha, h):
    # Create time np array from 0 to 10 with intervals of h

    # Solve Rk4 

    return 0

# Variable Definitions [alpha = 1, alpha = 2]
N_alpha_0 = [1e5, 1e5] # initial value of N_alpha
A_alpha = [0.1, 0.1]
B_alpha = [8e-7, 8e-7]
C_alpha = [1e-6, 1e-7]

# Loop through different step sizes (h) for t

Na = Rk4(alpha, h)
N1 = Rk4(1, h)
N2 = Rk4(2, h)

dNa_dt = Na(A_alpha - B_alpha * N_alpha)
birth_rate = A_alpha * Na
disease_death = B_alpha * Na ** 2
food_death = C_alpha * N1 * N2

# Plot N1(t) vs N2(t) with appropriate time step size

# Plot N1(10) and N2(10) logarithmic errors log10(1/h) for optimal step size vs increased by 2, 4, 8, and 16
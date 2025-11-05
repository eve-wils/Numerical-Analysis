# Evelyn Wilson
# Numerical Methods
# Lab 6 Question 1
# Due November 6

import numpy as np

# Variable Definitions [alpha = 1, alpha = 2]
N_alpha_0 = 1e5 # initial value of N_alpha
A = 0.1
B = 8e-7
C_alpha = [1e-6, 1e-7]

def rate_function(alpha, N1, N2):
    if(alpha == 0):
        return N1*(A - B*N1 - C_alpha[alpha]*N2)
    if(alpha == 1):
        return N2*(A - B*N2 - C_alpha[alpha]*N1)
# Function definitions
def Rk4(h):
    steps = int(10 / h + 1)
    Z = np.zeros((2, steps))
    Z[:,0] = N_alpha_0
    t = np.zeros((steps))
    t[0] = 0
    # Set the initial values as the first indices of each row
    for i in range(1, steps):
        t[i] = h * i
        omega_i = [Z[0, i-1], Z[1, i-1]]
        for j in range(2): # 0 (N_1) and 1 (N_2)
            # The rate function is independent of t, so we just evaluate as follows
            f1 = rate_function(j, omega_i[0], omega_i[1])
            f2 = rate_function(j, omega_i[0] + (h/2)*f1, omega_i[1] + (h/2)*f1)
            f3 = rate_function(j, omega_i[0] + (h/2)*f2, omega_i[1] + (h/2)*f2)
            f4 = rate_function(j, omega_i[0] + h*f3, omega_i[1] + h*f3)
            Z[j, i] = omega_i[j] + (h/6)*(f1 + 2*f2 + 2*f3 + f4)
    return Z, t

# Plotting
# Plot N1(t) vs N2(t) with appropriate time step size

import matplotlib.pyplot as plt
h_optimal = 10**-4
plt.figure(figsize=(10, 6))
Z, t = Rk4(h_optimal)
print(Z[:,-1])
plt.plot(t, Z[0,:], label='N1(t)')
plt.plot(t, Z[1,:], label='N2(t)')
plt.title('Population Dynamics: N1(t) and N2(t)')
plt.xlabel('Time (t)')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()

# Plot N1(10) and N2(10) logarithmic errors log10(1/h) for optimal step size vs increased by 2, 4, 8, and 16
h_values = [h_optimal * (2**i) for i in range(5)]
h_array = np.array(h_values)

errors_N1 = []
errors_N2 = []
N1_optimal = Z[0,-1]
N2_optimal = Z[1,-1]
for h in h_values:
    Z_new, _ = Rk4(h)
    errors_N1.append(np.log10(np.abs(Z_new[0,-1] - N1_optimal)))
    errors_N2.append(np.log10(np.abs(Z_new[1,-1] - N2_optimal)))
x = np.log10(1.0/h_array)
x_plot = x[1:]
y1_plot = np.array(errors_N1[1:])
y2_plot = np.array(errors_N2[1:])
plt.figure()
plt.plot(x_plot, y1_plot, label='N1(10) error')
plt.plot(x_plot, y2_plot, label='N2(10) error')
plt.title('log10(error) vs log10(1/h)')
plt.xlabel('log10(1/h)')
plt.ylabel('log10(|N_alpha(10) - reference|)')
plt.legend()
plt.grid(True)
plt.show()
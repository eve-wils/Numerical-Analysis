# Evelyn Wilson
# Numerical Methods
# Lab 6 Question 2
# Due November 6

import numpy as np
import matplotlib.pyplot as plt

def y(t, a, y0):
    return y0 * np.exp(-a * t)

# 2 step Adams Predictor/Corrector method
def adams_pc(a, h, y0, time):
    num_steps = int(time/h)
    t = np.zeros(num_steps+1)
    y = np.zeros(num_steps+1)
    y[0] = y0
    t[0] = 0.0
    
    # Exact solution for first step
    t[1] = h
    y[1] = y0 * np.exp(h * (-a))

    # AB2/AM2 Predictor-Corrector
    for i in range(1, num_steps):
        t[i+1] = t[i] + h

        # Function evaluations
        f_prev = (-a) * y[i-1]
        f_i = (-a) * y[i]

        # Predictor (AB2)
        y_pred = y[i] + h * (1.5 * f_i - 0.5 * f_prev)

        # Corrector (AM2) - single evaluation
        f_pred = (-a) * y_pred
        y[i+1] = y[i] + h * ((5.0/12.0) * f_pred + (2.0/3.0) * f_i - (1.0/12.0) * f_prev)

    return t, y

# Parameters
a = 1.0 # For a = 1, h < 2.4 for stability
y_0 = 50.0
tmax = 10.0

h_vals = [0.5, 1.5, 3.0]  # Different step sizes to test stability
''' 0.5 (stable), 1.5 (within stability region, larger error), 3.0 (unstable) '''
colors = ['tab:blue', 'tab:orange', 'tab:red']

# Compute exact curve for continuous comparison
t_exact = np.linspace(0, tmax, 500)
y_exact = y(t_exact, a, y_0)

# Plotting
plt.plot(t_exact, y_exact, 'k-', linewidth=2, label='Analytic $y=50e^{-at}$')

# Loop over all h values and plot each numerical result
for h, c in zip(h_vals, colors):
    t_num, y_num = adams_pc(a, h, y_0, tmax)
    plt.plot(t_num, y_num, '--o', color=c, label=f'AB2–AM2 PC (h={h})')

plt.title(f'Adams 2-Step Predictor–Corrector Stability Demonstration (a={a})')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.show()

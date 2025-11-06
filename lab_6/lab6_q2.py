# Evelyn Wilson
# Numerical Methods
# Lab 6 Question 2
# Due November 6

import numpy as np

# 2 step Adams Predictor/Corrector method
def adams_pc(a, h, y0, time):
    num_steps = int(time/h + 1)
    t = np.zeros(num_steps+1)
    y = np.zeros(num_steps)
    y[0] = y0
    t[0] = 0.0
    
    # Exact solution for first step
    t[1] = 1
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
# Verify analysis by programming the predictor/corrector scheme solving the ODE
# Use several different step sizes, one which violates the stability criterion,
# another which satisfies and a third which satisfies but is not very accurate

y_0 = 50.0
h = None

# Plot your solution as a function of time in each chase thereby providing graphical evidence of either stability or instability


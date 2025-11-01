# Evelyn Wilson
# Numerical Methods
# Lab 6 Question 2
# Due November 6

# Analyze stability of the equation

def dy_dt(t):
    return - a * y(t)

''' Determine the maximum allowable step size that will still maintain stability for each method '''

# 2nd order Runge-Kutta method (Midpoint rule)

# Adams-Bashforth 2 step method

# Adams Moulton 2 step method

# 2 step Adams Predictor/Corrector method

# Verify analysis by programming the predictor/corrector scheme solving the ODE
# Use several different step sizes, one which violates the stability criterion,
# another which satisfies and a third which satisfies but is not very accurate

y_0 = 50.0

# Plot your solution as a function of time in each chase thereby providing graphical evidence of either stability or instability


# Evelyn Wilson
# Numerical Methods
# Lab 6 Question 3
# Due November 6

# Retry executing the Van der Pol AB4-AM3 simulation and plotting.
import numpy as np
import matplotlib.pyplot as plt

def van_der_pol(u, a):
    y, v = u
    dy = v
    dv = a * v - v**3 - y
    return np.array([dy, dv])

def rk4_step(w_i, h, a):
    f1 = van_der_pol(w_i, a)
    f2 = van_der_pol(w_i + 0.5*h*f1, a)
    f3 = van_der_pol(w_i + 0.5*h*f2, a)
    f4 = van_der_pol(w_i + h*f3, a)
    return w_i + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)

def solve_vdp_ab4_am3(a, h=0.01, tmax=100.0):
    num_steps = int(np.ceil(tmax / h))
    t = np.linspace(0, num_steps*h, num_steps+1)
    w = np.zeros((num_steps+1, 2))
    w[0, :] = np.array([0.0, 0.1])
    # seed with RK4 for first three steps
    for n in range(0, 3):
        w[n+1, :] = rk4_step(w[n, :], h, a)
    F = np.zeros_like(w)
    for n in range(0, 4):
        F[n, :] = van_der_pol(w[n, :], a)
    for n in range(3, num_steps):
        w_pred = w[n, :] + (h/24.0) * (55.0*F[n, :] - 59.0*F[n-1, :] + 37.0*F[n-2, :] - 9.0*F[n-3, :])
        F_pred = van_der_pol(w_pred, a)
        w[n+1, :] = w[n, :] + h * ((9.0/24.0)*F_pred + (19.0/24.0)*F[n, :] - (5.0/24.0)*F[n-1, :] + (1.0/24.0)*F[n-2, :])
        F[n+1, :] = van_der_pol(w[n+1, :], a)
    return t, w

# Parameters
a_values = [0.5, 1.5, 2.5, 3.5, 4.5]
h = 0.01
tmax = 100.0

plt.figure(figsize=(8, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

for a, c in zip(a_values, colors):
    t, U = solve_vdp_ab4_am3(a, h=h, tmax=tmax)
    plt.plot(U[:,0], U[:,1], color=c, linewidth=1.0, label=f'a={a}')

plt.xlabel('y (voltage)')
plt.ylabel("y' (current)")
plt.title(f'Van der Pol A-B/A-M 4-step corrector, h={h}, t in [0,{tmax}]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

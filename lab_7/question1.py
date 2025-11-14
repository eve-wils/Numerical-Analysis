# Evelyn Wilson
# Lab 7, Question 1
# Due Date: November 14, 2025

import numpy as np
import matplotlib.pyplot as plt

# --- Problem Constants ---
# Per the problem description 
L = 50.0       # length (in)
D = 8.5e7      # flexual rigidity (lb in)
S = 100.0      # axial force (lb in^-1)
q = 1000.0     # uniform load (lb in^-2)

# --- ODE System Definition ---

def beam_ode(x, z, S, D, q, L):
    z1, z2 = z
    
    # Calculate terms from the original equation 
    term1 = (S / D) * z2
    term2 = (q * x * (x - L)) / (2 * D) * z1
    term3 = (1.0 + z2**2)**1.5
    
    # Calculate derivatives
    dz1_dx = z2
    dz2_dx = (term1 + term2) * term3

    return np.array([dz1_dx, dz2_dx])

# ODE Solver (Predictor-Corrector)

# Rk4 to calculate initial steps
def rk4_step(ode, x_i, w_i, h, **f_args): 
    f1 = ode(x_i, w_i, **f_args)
    f2 = ode(x_i + 0.5*h, w_i + 0.5*h*f1, **f_args)
    f3 = ode(x_i + 0.5*h, w_i + 0.5*h*f2, **f_args)
    f4 = ode(x_i + h, w_i + h*f3, **f_args)

    return w_i + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)

def ab4_am3(ode, x_span, w0, h, **f_args):
    t0, tmax = x_span
    num_steps = int(np.ceil((tmax - t0) / h))
    t = np.linspace(t0, t0 + num_steps*h, num_steps+1)
    
    # Ensure w0 is a 1D array
    w0 = np.asarray(w0)
    w = np.zeros((num_steps+1, w0.shape[0]))
    w[0, :] = w0
    
    # Seed with RK4 for the first three steps (to get 4 points: 0, 1, 2, 3)
    for n in range(0, 3):
        w[n+1, :] = rk4_step(ode, t[n], w[n, :], h, **f_args)
    # Pre-calculate F values for the first 4 points
    F = np.zeros_like(w)
    for n in range(0, 4):
        F[n, :] = ode(t[n], w[n, :], **f_args)
    
    # Main predictor-corrector loop
    for n in range(3, num_steps):
        # AB4 Predictor
        w_pred = w[n, :] + (h/24.0) * (55.0*F[n, :] - 59.0*F[n-1, :] + 37.0*F[n-2, :] - 9.0*F[n-3, :])
        # Evaluate F at predicted step
        F_pred = ode(t[n+1], w_pred, **f_args)
        # AM3 Corrector
        w[n+1, :] = w[n, :] + h/24.0 * (9.0*F_pred + 19.0*F[n, :] - 5.0*F[n-1, :] + 1.0*F[n-2, :])
        # Evaluate F at the corrected step for the next iteration
        F[n+1, :] = ode(t[n+1], w[n+1, :], **f_args)
        
    return t, w

# Begin Shooting Method

def shooting_func(gamma, h):
    w0 = np.array([0.0, gamma])  # Initial conditions: y(0)=0, y'(0)=gamma
    x_span = (0.0, L)
    
    # Solve the BVP as an IVP
    x, w = ab4_am3(beam_ode, x_span, w0, h, S=S, D=D, q=q, L=L)
    
    # Return the error at x=L (the value of y(L), which should be 0)
    y_at_L = w[-1, 0]
    return y_at_L

def secant_method(f, u0, u1, h, tol=1e-8, max_iter=50):
    f0 = f(u0, h)
    f1 = f(u1, h)
    print(f"u_0: u={u0:<12.7e}, F(u) (y(L))={f0:<12.7e}")
    print(f"  Guess 1: u={u1:<12.7e}, F(u) (y(L))={f1:<12.7e}")

    for n in range(2, max_iter + 1):
        # Check for division by zero
        if abs(f1 - f0) < 1e-15:
            return u1
            
        # Secant formula
        u_new = u1 - f1 * (u1 - u0) / (f1 - f0)
        f_new = f(u_new, h)

        print(f"Guess {n}: u={u_new:<12.7e}, F(u) (y(L))={f_new:<12.7e}")

        # Check for convergence
        if abs(f_new) < tol:
            print(f"\nConvergence reached. Final u = {u_new:.7e}")
            return u_new

        # Update for next iteration
        u0, f0 = u1, f1
        u1, f1 = u_new, f_new
    return None

u_guess_0 = -0.1
u_guess_1 = -0.05

step_sizes_to_test = [1.0, 0.5, 0.25]

print("Solving Problem 1: Nonlinear Beam Deflection")

for h in step_sizes_to_test:
    print(f"step size h = {h}")
    
    # Find the correct initial slope u using the Secant method
    final_u = secant_method(shooting_func, u_guess_0, u_guess_1, h=h, tol=1e-8, max_iter=30)

    if final_u is not None:
        # Solve one last time with the converged u to get the solution
        x_sol, w_sol = ab4_am3(beam_ode, (0, L), np.array([0.0, final_u]), h, S=S, D=D, q=q, L=L)
        
        y_sol = w_sol[:, 0]
        y_prime_sol = w_sol[:, 1]
        
        # Report final values
        print(f"Final converged u (y'(0)): {final_u:.7e}")
        print(f"Final computed y(L): {y_sol[-1]:.7e}")

        # Plot Deflection y(x)
        plt.plot(x_sol, y_sol, label=f'$y(x)$, $h={h}$')
        plt.title(f'Beam Deflection $y(x)$ for Step Size $h={h}$')
        plt.xlabel('Position $x$ (in)')
        plt.ylabel('Deflection $y$ (in)')
        plt.legend()
deflection_filename = f'beam_deflection.png'
plt.savefig(deflection_filename)
plt.close()

for h in step_sizes_to_test:
    print(f"step size h = {h}")
    
    # Find the correct initial slope u using the Secant method
    final_u = secant_method(shooting_func, u_guess_0, u_guess_1, h=h, tol=1e-8, max_iter=30)

    if final_u is not None:
        # Plot Slope y'(x) 
        plt.figure(figsize=(10, 6))
        plt.plot(x_sol, y_prime_sol, label=f"$y'(x)$, $h={h}$", color='orange')
        plt.title(f"Beam Slope $y'(x)$ for Step Size $h={h}$")
        plt.xlabel('Position $x$ (in)')
        plt.ylabel("Slope $y'$ (in/in)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    else:
        print(f"  Solution failed to converge for h={h}")

filename = f'beam_slope.png'
plt.savefig(filename)
plt.close()
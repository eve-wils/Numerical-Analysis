# Evelyn Wilson
# Lab 7, Question 2a and b
# Due Date: November 14, 2025
# Using the same file unlike normal because there are a lot of dependencies.
import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 1.0           # Length (m)
lambda_sq = 2.7   # m/kappa (m^-2)
lambda_val = np.sqrt(lambda_sq)
Tc = 37.0         # Core temp (°C)
Ts = 32.0         # Surface temp (°C)
Ta = Tc           # Arteriole temp (°C)

# Build A matrix for FDM
def build_matrix(N, dx, lambda_sq):

    # Main diagonal value
    diag_val = -(2.0 + lambda_sq * dx**2)
    
    # Create diagonals
    main_diag = np.full(N, diag_val)
    off_diag = np.ones(N - 1)
    
    # Assemble the matrix
    A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    return A

def analytic_solution_2a(x, L, Ts, Tc, lambda_val):
   
    T_tilde_L = Ts - Tc
    return T_tilde_L * np.sinh(lambda_val * x) / np.sinh(lambda_val * L)

def solve_problem_2a():

    # Boundary conditions for T-tilde
    T_tilde_0 = Tc - Ta  # This is 0
    T_tilde_L = Ts - Tc  # 

    # N+1 values to test
    segment_counts = np.array([5, 10, 20, 40, 80, 160])
    errors = []
    
    # Store solution for N+1 = 20 for plotting
    x_plot = None
    T_tilde_plot = None
    analytic_plot = None

    for num_segments in segment_counts:
        N = num_segments - 1  # Number of interior nodes
        dx = L / num_segments # Node spacing
        x_nodes = np.linspace(0, L, num_segments + 1)
        # Build matrix A and vector b
        A = build_matrix(N, dx, lambda_sq)
        b = np.zeros(N)
        # Apply boundary conditions to b vector
        b[0] = -T_tilde_0   # b[0] = 0
        b[-1] = -T_tilde_L
        
        # Solve using left division
        T_tilde_interior = np.linalg.solve(A, b)
        T_tilde_full = np.concatenate(([T_tilde_0], T_tilde_interior, [T_tilde_L]))
        #Error analysis
        T_analytic = analytic_solution_2a(x_nodes, L, Ts, Tc, lambda_val)
        max_err = np.max(np.abs(T_tilde_full - T_analytic))
        errors.append(max_err)
        
        if num_segments == 20:
            x_plot = x_nodes
            T_tilde_plot = T_tilde_full
            analytic_plot = T_analytic

    # Plotting 2a
    
    # T-tilde vs x
    plt.figure()
    plt.plot(x_plot, T_tilde_plot, 'bo', label='Numerical (N+1=20)', markersize=5)
    plt.plot(x_plot, analytic_plot, 'r-', label='Analytic Solution')
    plt.xlabel('$x$ (m)')
    plt.ylabel('$\tilde{T} = T - T_a$ (°C)')
    plt.title('Problem 2a: Steady-State Temperature (No Heating)')
    plt.legend()
    plt.grid(True)
    
    # Error vs N+1 (log-log)
    plt.figure()
    plt.loglog(segment_counts, errors, 'ko-', label='Max Error $\epsilon$')
    
    # Add reference line for second order convergence
    # Error ~ C * (dx)^2 ~ C * (1/(N+1))^2
    C = errors[0] * (segment_counts[0]**2)
    plt.loglog(segment_counts, C / (segment_counts**2), 'r--', label='Slope = -2 (2nd Order)')
    
    plt.xlabel('Num Segments ($N+1$)')
    plt.ylabel('Max Error $\epsilon$')
    plt.title('Problem 2a: Error Analysis (Convergence)')
    plt.legend()
    plt.grid(True)

# Problem 2b

def problem_2b(num_segments, L, lambda_sq, Tc, Ts, Ta, sigma_E_kappa, gamma):
    
    # Boundary conditions for T-tilde
    T_tilde_0 = Tc - Ta  # 0
    T_tilde_L = Ts - Tc
    
    N = num_segments - 1
    dx = L / num_segments
    x_nodes = np.linspace(0, L, num_segments + 1)
    x_interior = x_nodes[1:-1] # Interior node positions
    
    # Build A matrix
    A = build_matrix(N, dx, lambda_sq)
    
    # Source term S(x) = (sigma*E0/kappa) * exp(-gamma*(L-x))
    source_term = sigma_E_kappa * np.exp(-gamma * (L - x_interior))
    
    # Build b vector
    b = -dx**2 * source_term

    # Apply BCs
    b[0] -= T_tilde_0
    b[-1] -= T_tilde_L
    
    # solve using left division
    T_tilde_interior = np.linalg.solve(A, b)
    
    # Assemble T-tilde and convert back to T
    T_tilde_full = np.concatenate(([T_tilde_0], T_tilde_interior, [T_tilde_L]))
    T_full = T_tilde_full + Ta
    
    return x_nodes, T_full
    
def solve_problem_2b():
    
    # constants for 2b
    sigma_E_kappa = 100.0  # (sigma*E0 / kappa)
    gamma = 1.0 / L        # (L^-1)

    # N+1 values to test
    segment_counts = np.array([5, 10, 20, 40, 80, 160])
    errors = []
    
    # Store solution for N+1 = 20 for plotting
    x_plot = None
    T_plot = None
    
    # Error Analysis (vs. High-Fidelity Solution)
    x_truth, T_truth = problem_2b(640, L, lambda_sq, Tc, Ts, Ta, sigma_E_kappa, gamma)

    for num_segments in segment_counts:
        x_n, T_n = problem_2b(num_segments, L, lambda_sq, Tc, Ts, Ta, sigma_E_kappa, gamma)

        # Interpolate the "truth" solution onto the coarser grid (x_n)
        T_truth_interp = np.interp(x_n, x_truth, T_truth)
        
        # Calculate max error
        max_err = np.max(np.abs(T_n - T_truth_interp))
        errors.append(max_err)
        
        # Save the N+1=20 case for plotting
        if num_segments == 20:
            x_plot = x_n
            T_plot = T_n

    # Plotting 2b
    
    # T vs x
    plt.figure()
    plt.plot(x_plot, T_plot, 'b-', label='Numerical $T(x)$ (N+1=20)')
    plt.axhline(y=Ta, color='r', linestyle='--', label=f'Arteriole Temp $T_a = {Ta:.1f}$°C')
    plt.xlabel('$x$ (m)')
    plt.ylabel('Tissue Temperature $T$ (°C)')
    plt.title('Problem 2b: Steady-State Temperature (With Heating)')
    plt.legend()
    plt.grid(True)
    
    # Error vs N+1 (log-log)
    plt.figure()
    plt.loglog(segment_counts, errors, 'ko-', label='Max Error $\epsilon$ (vs. truth)')
    
    # Add second-order reference line to compare convergence.
    C = errors[0] * (segment_counts[0]**2)
    plt.loglog(segment_counts, C / (segment_counts**2), 'r--', label='Slope = -2 (2nd Order)')
    
    plt.xlabel('Number of Segments ($N+1$)')
    plt.ylabel('Max Error $\epsilon$')
    plt.title('Problem 2b: Error Analysis (Convergence)')
    plt.legend()
    plt.grid(True)

solve_problem_2a()
solve_problem_2b()

plt.show()
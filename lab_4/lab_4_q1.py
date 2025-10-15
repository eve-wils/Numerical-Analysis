import numpy as np
import matplotlib.pyplot as plt

# Input data
x_data = np.array([0, 4, 6, 7, 13, 15, 18, 21, 22, 27, 29, 32, 35, 36, 37, 42, 47, 50])
y_data = np.array([9.9096816e+03, 8.0288293e+02, 3.2994196e+02, 1.2437454e+02, 
                   4.7990217e+00, 1.2273452e+00, 1.3521287e-01, 1.8341061e-02,
                   1.4652968e-02, 4.9403019e-04, 1.7287294e-04, 2.2708755e-05,
                   3.2690255e-06, 1.3780727e-06, 8.2726513e-07, 3.4272876e-08,
                   1.6249468e-09, 2.8535191e-10])

### LINEAR POLYNOMIAL FIT ###
n = len(x_data)

# Construct design matrix A for linear fit
A_linear = np.zeros((n, 2))
A_linear[:, 0] = 1           # Column of ones for a0
A_linear[:, 1] = x_data      # x values for a1

# Compute A transverse * A (the normal matrix)
ATA_linear = A_linear.T @ A_linear

# Compute A^T * y (the right-hand side)
ATy_linear = A_linear.T @ y_data

# Solve the normal equations: (A^T * A) * c = A^T * y
linear_coeffs = np.linalg.solve(ATA_linear, ATy_linear)
a0_linear, a1_linear = linear_coeffs

print(f"Linear fit equation: y = {a0_linear:.4e} + {a1_linear:.4e}*x")

# Generate fitted values for plotting
y_linear_fit = a0_linear + a1_linear * x_data

### CUBIC POLYNOMIAL FIT ###

A_cubic = np.zeros((n, 4))
A_cubic[:, 0] = 1           # Column of ones for a0
A_cubic[:, 1] = x_data      # x values for a1
A_cubic[:, 2] = x_data**2   # x^2 values for a2
A_cubic[:, 3] = x_data**3   # x^3 values for a3

# Compute A^T * A (the normal matrix)
ATA_cubic = A_cubic.T @ A_cubic

# Compute A^T * y (the right-hand side)
ATy_cubic = A_cubic.T @ y_data

# Solve the normal equations
cubic_coeffs = np.linalg.solve(ATA_cubic, ATy_cubic)
a0_cubic, a1_cubic, a2_cubic, a3_cubic = cubic_coeffs

print(f"\nCubic fit equation:")
print(f"y = {a0_cubic:.4e} + {a1_cubic:.4e}*x + {a2_cubic:.4e}*x^2 + {a3_cubic:.4e}*x^3")

# Fitted values
y_cubic_fit = a0_cubic + a1_cubic*x_data + a2_cubic*x_data**2 + a3_cubic*x_data**3

### LINEARIZED FIT ###

# Take natural log of data
ln_y_data = np.log(y_data)

# Perform manual linear fit on transformed data: ln(y) = ln(A) + B*x

A_linearized = np.zeros((n, 2))
A_linearized[:, 0] = 1           # Column of ones for ln(A)
A_linearized[:, 1] = x_data      # x values for B

# Compute A^T * A
ATA_linearized = A_linearized.T @ A_linearized

# Compute A^T * ln(y)
ATy_linearized = A_linearized.T @ ln_y_data

# Solve the normal equations
linearized_coeffs = np.linalg.solve(ATA_linearized, ATy_linearized)
ln_A, B = linearized_coeffs
A_exp = np.exp(ln_A)  # Undo natural log

print(f"Assumed model: y = A * exp(B*x)")
print(f"After linearization: ln(y) = ln(A) + B*x")
print(f"\nResults: A = {A_exp:.4e}, B = {B:.4e}")
print(f"Therefore: y = {A_exp:.4e} * exp({B:.4e}*x)")

# Create linearized fit
y_linearized_fit = A_exp * np.exp(B * x_data)

# ============================================================================
# METHOD 4: MANUAL NON-LINEAR LEAST SQUARES FIT (Direct exponential fit)
# ============================================================================
print("\n" + "=" * 70)
print("METHOD 4: MANUAL NON-LINEAR LEAST SQUARES FIT")
print("=" * 70)

# For non-linear fit, we use Gauss-Newton method to minimize:
# sum((y_i - A*exp(B*x_i))^2)

# Define the exponential model
def exponential_model(x, A, B):
    """Exponential decay model: y = A * exp(B*x)"""
    return A * np.exp(B * x)

# Define the residual function
def residuals(params, x, y):
    """Calculate residuals: y_data - y_model"""
    A, B = params
    return y - exponential_model(x, A, B)

# Define the Jacobian matrix for the exponential model
def jacobian(params, x):
    """
    Jacobian matrix of the exponential model
    Partial derivatives: df/dA = exp(B*x), df/dB = A*x*exp(B*x)
    """
    A, B = params
    J = np.zeros((len(x), 2))
    exp_term = np.exp(B * x)
    J[:, 0] = exp_term           # df/dA
    J[:, 1] = A * x * exp_term   # df/dB
    return J

# Gauss-Newton algorithm
# Start with linearized solution as initial guess
params = np.array([A_exp, B])

max_iterations = 100
tolerance = 1e-10

for iteration in range(max_iterations):
    # Calculate residuals
    r = residuals(params, x_data, y_data)
    
    # Calculate Jacobian
    J = jacobian(params, x_data)
    
    # Gauss-Newton update: (J^T * J) * delta = J^T * r
    JTJ = J.T @ J
    JTr = J.T @ r
    
    # Solve for parameter update
    delta = np.linalg.solve(JTJ, JTr)
    
    # Update parameters
    params = params + delta
    
    # Check for convergence
    if np.linalg.norm(delta) < tolerance:
        break

A_nonlinear, B_nonlinear = params

print(f"Non-linear least squares results:")
print(f"  A = {A_nonlinear:.4e}, B = {B_nonlinear:.4e}")
print(f"  Converged in {iteration + 1} iterations")

# Generate fitted values
y_nonlinear_fit = exponential_model(x_data, A_nonlinear, B_nonlinear)

# ============================================================================
# COMPARISON: Linearized vs Non-linear Fit
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON: Linearized vs Non-linear Exponential Fit")
print("=" * 70)

print(f"\nParameter comparison:")
print(f"  Linearized:  A = {A_exp:.4e}, B = {B:.4e}")
print(f"  Non-linear:  A = {A_nonlinear:.4e}, B = {B_nonlinear:.4e}")

# Calculate residuals (errors) for each method
residuals_linearized = y_data - y_linearized_fit
residuals_nonlinear = y_data - y_nonlinear_fit

# Calculate sum of squared errors
SSE_linearized = np.sum(residuals_linearized**2)
SSE_nonlinear = np.sum(residuals_nonlinear**2)

print(f"\nSum of Squared Errors:")
print(f"  Linearized: {SSE_linearized:.4e}")
print(f"  Non-linear: {SSE_nonlinear:.4e}")

# ============================================================================
# PLOTTING: Linear and Semilog Scales
# ============================================================================
plt.figure(figsize=(14, 10))

# Plot 1: All fits on linear scale
plt.subplot(2, 2, 1)
plt.plot(x_data, y_data, 'ko', markersize=8, label='Experimental Data', zorder=5)
plt.plot(x_data, y_linear_fit, 'b-', linewidth=2, label='Linear Fit')
plt.plot(x_data, y_cubic_fit, 'r-', linewidth=2, label='Cubic Fit')
plt.plot(x_data, y_linearized_fit, 'g--', linewidth=2, label='Linearized Exp Fit')
plt.plot(x_data, y_nonlinear_fit, 'm:', linewidth=3, label='Non-linear Exp Fit')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('All Fits - Linear Scale', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 2: Exponential fits comparison on linear scale
plt.subplot(2, 2, 2)
plt.plot(x_data, y_data, 'ko', markersize=8, label='Experimental Data', zorder=5)
plt.plot(x_data, y_linearized_fit, 'g--', linewidth=2, label='Linearized Exp Fit')
plt.plot(x_data, y_nonlinear_fit, 'm:', linewidth=3, label='Non-linear Exp Fit')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Exponential Fits - Linear Scale', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 3: All fits on semilog scale
plt.subplot(2, 2, 3)
plt.semilogy(x_data, y_data, 'ko', markersize=8, label='Experimental Data', zorder=5)
plt.semilogy(x_data, y_linear_fit, 'b-', linewidth=2, label='Linear Fit')
plt.semilogy(x_data, y_cubic_fit, 'r-', linewidth=2, label='Cubic Fit')
plt.semilogy(x_data, y_linearized_fit, 'g--', linewidth=2, label='Linearized Exp Fit')
plt.semilogy(x_data, y_nonlinear_fit, 'm:', linewidth=3, label='Non-linear Exp Fit')
plt.xlabel('x', fontsize=12)
plt.ylabel('y (log scale)', fontsize=12)
plt.title('All Fits - Semilog Scale', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, which='both')

# Plot 4: Residuals comparison on log scale
plt.subplot(2, 2, 4)
plt.semilogy(x_data, np.abs(residuals_linearized), 'go-', linewidth=2, 
             markersize=6, label='Linearized Fit')
plt.semilogy(x_data, np.abs(residuals_nonlinear), 'mo-', linewidth=2, 
             markersize=6, label='Non-linear Fit')
plt.xlabel('x', fontsize=12)
plt.ylabel('Absolute Error (log scale)', fontsize=12)
plt.title('Residuals Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('lab4_q1_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# ANSWER TO ADVISOR'S QUESTION
# ============================================================================
print("\n" + "=" * 70)
print("ANSWER TO ADVISOR'S QUESTION")
print("=" * 70)
print("\nWhy didn't we initially do a non-linear least-squares fit?")
print("-" * 70)
print("The linearized approach is simpler and provides good initial estimates.")
print("\nDo they give the same answer?")
print("-" * 70)
print("The methods give slightly different results because:")
print("  - Linearized fit minimizes error in ln(y) space")
print("  - Non-linear fit minimizes error in y space")
print(f"\nIn this case, non-linear SSE ({SSE_nonlinear:.4e}) is")
print(f"{'smaller' if SSE_nonlinear < SSE_linearized else 'larger'} than linearized SSE ({SSE_linearized:.4e})")
print("The non-linear fit is more accurate for the original data.")
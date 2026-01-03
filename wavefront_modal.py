# This file contains all mathematical operations needed to reconstruct a wavefront from Shack–Hartmann spot displacements using Zernike polynomials.
# It converts centroid shifts into slopes, builds and solves the modal least-squares system, reconstructs the wavefront on a grid, removes piston and tilt, and computes PV/RMS
# Caching and vectorized operations are used to speed up repeated calculations.

import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from backend import xp, GPU_ENABLED
# Cache used to store precomputed Zernike grids for reuse
_ZERNIKE_GRID_CACHE = {}
# Try to import numba for optional acceleration

# This function normalizes centroid coordinates to a unit pupil
def normalize_coordinates(centroids):
    # Convert input to NumPy or GPU array
    centroids = xp.asarray(centroids)
    # Separate x and y coordinates
    x = centroids[:, 0]
    y = centroids[:, 1]
    # Center the coordinates around the pupil center
    x = x - xp.mean(x)
    y = y - xp.mean(y)
    # Compute radial distance from the center
    r = xp.sqrt(x**2 + y**2)
    # Find the maximum radius
    r_max = xp.max(r)
    # Normalize coordinates so the pupil radius is 1
    x_norm = x / r_max
    y_norm = y / r_max
    # Keep only points inside the unit pupil
    mask = (x_norm**2 + y_norm**2) <= 1.0
    return x_norm[mask], y_norm[mask], mask

# This function extracts x and y slopes from displacement vectors
def compute_slopes(displacements, mask):
    # Convert displacements to NumPy or GPU array
    displacements = xp.asarray(displacements)
    # Extract x-direction slopes for valid points
    dx = displacements[:, 0][mask]
    # Extract y-direction slopes for valid points
    dy = displacements[:, 1][mask]
    return dx, dy

# This function converts Cartesian coordinates to polar form
def cartesian_to_polar(x, y):
    # Convert inputs to NumPy or GPU arrays
    x = xp.asarray(x)
    y = xp.asarray(y)
    # Compute radial distance from the origin
    rho = xp.sqrt(x**2 + y**2)
    # Compute angle relative to the x-axis
    theta = xp.arctan2(y, x)
    return rho, theta

# This function computes the radial part of a Zernike polynomial
def zernike_radial(n, m, rho):
    # Ensure input is a NumPy or GPU array
    rho = xp.asarray(rho)
    # Initialize the radial polynomial values
    R = xp.zeros_like(rho)
    # Compute the radial polynomial using the standard Zernike formula
    for k in range((n - abs(m)) // 2 + 1):
        num = (-1)**k * math.factorial(n - k)
        den = (
            math.factorial(k)
            * math.factorial((n + abs(m)) // 2 - k)
            * math.factorial((n - abs(m)) // 2 - k)
        )
        R = R + (num / den) * rho ** (n - 2*k)
    return R

# This function computes the full Zernike polynomial using radius and angle
def zernike(n, m, rho, theta):
    # Ensure inputs are NumPy or GPU arrays
    rho = xp.asarray(rho)
    theta = xp.asarray(theta)
    # Compute the radial component
    R = zernike_radial(n, m, rho)
    # Apply angular dependence based on the sign of m
    if m > 0:
        return R * xp.cos(m * theta)
    elif m < 0:
        return R * xp.sin(-m * theta)
    else:
        return R

# This function numerically estimates partial derivatives using finite differences
def numerical_derivative(func, x, y, eps=1e-6):
    # Ensure inputs are NumPy or GPU arrays
    x = xp.asarray(x)
    y = xp.asarray(y)
    # Small step size used for differentiation
    eps_xp = xp.asarray(eps)
    # Estimate derivative with respect to x
    f_x1 = func(x + eps_xp, y)
    f_x2 = func(x - eps_xp, y)
    dfdx = (f_x1 - f_x2) / (2 * eps_xp)
    # Estimate derivative with respect to y
    f_y1 = func(x, y + eps_xp)
    f_y2 = func(x, y - eps_xp)
    dfdy = (f_y1 - f_y2) / (2 * eps_xp)
    return dfdx, dfdy

# This function computes x and y derivatives of a Zernike mode
def zernike_derivatives(n, m, x, y):
    # Convert inputs to NumPy arrays for numerical differentiation
    x = np.asarray(x)
    y = np.asarray(y)
    # Define the Zernike function in Cartesian coordinates
    def Z(xi, yi):
        # Convert x,y to polar coordinates
        rho, theta = cartesian_to_polar(xi, yi)
        # Evaluate the Zernike polynomial at those points
        return zernike(n, m, rho, theta)
    # Compute numerical derivatives with respect to x and y
    dZdx, dZdy = numerical_derivative(Z, x, y)
    return dZdx, dZdy
    # (The line below is unreachable and has no effect)
    return numerical_derivative(Z, x, y)

# This function generates a list of Zernike mode indices (n, m)
def generate_zernike_modes(n_modes, include_piston=False):
    modes = []
    # Optionally include the piston mode (n=0, m=0)
    if include_piston:
        modes.append((0, 0))
    n = 1
    # Generate modes in increasing radial order
    while len(modes) < n_modes:
        for m in range(-n, n + 1, 2):
            if len(modes) >= n_modes:
                break
            modes.append((n, m))
        n += 1
    return modes

# This function normalizes centroid positions to a unit pupil
def normalize_centroids(centroids):
    # Convert input to NumPy or GPU array
    centroids = xp.asarray(centroids)
    # Compute the center of the centroid cloud
    cx = xp.mean(centroids[:, 0])
    cy = xp.mean(centroids[:, 1])
    # Shift centroids so the pupil is centered at the origin
    x = centroids[:, 0] - cx
    y = centroids[:, 1] - cy
    # Compute distance of each point from the center
    r = xp.sqrt(x**2 + y**2)
    # Find the pupil radius in pixels
    R_px = xp.max(r)
    # Normalize coordinates so pupil radius becomes 1
    x_norm = x / R_px
    y_norm = y / R_px
    return x_norm, y_norm, R_px

# This function builds the linear system that links Zernike coefficients to measured slopes
def build_design_matrix(x, y, modes, R_phys):
    # Force computation on CPU using standard NumPy for numerical stability
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    R_phys = float(R_phys)
    # Number of valid centroid locations
    N = x.shape[0]
    # Number of Zernike modes
    M = len(modes)
    # Create the design matrix with one row per slope component
    A = np.zeros((2 * N, M), dtype=np.float64)
    # Fill each column with derivatives of one Zernike mode
    for j, (n, m) in enumerate(modes):
        # Compute partial derivatives of the Zernike mode
        dZdx, dZdy = zernike_derivatives(n, m, x, y)
        # X-slopes go in even rows, Y-slopes go in odd rows
        A[0::2, j] = dZdx / R_phys
        A[1::2, j] = dZdy / R_phys
    return A

# This function stacks x and y displacements into a single slope vector
def build_slope_vector(displacements):
    # Convert input to NumPy or GPU array
    displacements = xp.asarray(displacements)
    # Number of measured spot displacements
    N = displacements.shape[0]
    # Create a vector twice as long to hold x and y components
    b = xp.zeros(2 * N, dtype=displacements.dtype)
    # Insert x-displacements in even entries
    b[0::2] = displacements[:, 0]
    # Insert y-displacements in odd entries
    b[1::2] = displacements[:, 1]
    return b

# This function solves for Zernike coefficients using least squares
def solve_modal_coefficients(A, b):
    # Convert inputs to NumPy or GPU arrays
    A = xp.asarray(A)
    b = xp.asarray(b)
    # Solve the overdetermined system in a least-squares sense
    C, residuals, rank, s = xp.linalg.lstsq(A, b, rcond=None)
    return C

# This function checks whether a given (n, m) pair is a valid Zernike mode
def is_valid_zernike(n, m):
    return (
        n >= 0 and
        abs(m) <= n and
        (n - abs(m)) % 2 == 0
    )

# This function reconstructs the wavefront from Zernike coefficients
def reconstruct_wavefront(coeffs, modes, grid_size=200):
    # Ensure coefficients are valid numeric values
    coeffs = np.asarray(coeffs, dtype=np.float64)
    # Keep only finite coefficients with valid Zernike modes
    valid = [
        (c, (n, m))
        for c, (n, m) in zip(coeffs, modes)
        if np.isfinite(c) and is_valid_zernike(n, m)
    ]
    # Stop if no valid modes are available
    if not valid:
        raise ValueError("No valid Zernike modes passed to reconstruct_wavefront()")
    # Separate filtered coefficients and modes
    coeffs, modes = zip(*valid)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    modes = tuple(modes)
    M = len(coeffs)
    # Use grid size and modes as a cache key
    cache_key = (grid_size, modes)
    # Reuse precomputed grids if available
    if cache_key in _ZERNIKE_GRID_CACHE:
        X, Y, pupil, Z_stack = _ZERNIKE_GRID_CACHE[cache_key]
    else:
        # Create a square grid covering the unit pupil
        x = np.linspace(-1.0, 1.0, grid_size)
        y = np.linspace(-1.0, 1.0, grid_size)
        X, Y = np.meshgrid(x, y, indexing="xy")
        # Convert grid to polar coordinates
        rho = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        # Define the circular pupil region
        pupil = rho <= 1.0
        # Precompute all Zernike modes on the grid
        Z_stack = np.zeros((M, grid_size, grid_size), dtype=np.float64)
        for j, (n, m) in enumerate(modes):
            Zj = np.zeros_like(rho)
            Zj[pupil] = zernike(n, m, rho[pupil], theta[pupil])
            Z_stack[j] = Zj
        # Cache the grid and Zernike values for reuse
        _ZERNIKE_GRID_CACHE[cache_key] = (X, Y, pupil, Z_stack)
    # Combine Zernike modes using their coefficients
    W = np.tensordot(coeffs, Z_stack, axes=(0, 0))
    # Mask values outside the pupil
    W[~pupil] = np.nan
    return X, Y, W

# This function removes piston and tilt from a reconstructed wavefront
def remove_piston_tilt(W, X, Y):
    # Convert inputs to NumPy or GPU arrays
    W = xp.asarray(W)
    X = xp.asarray(X)
    Y = xp.asarray(Y)
    # Identify valid wavefront values
    valid = xp.isfinite(W)
    # Return early if there are too few valid points
    if xp.sum(valid) < 3:
        return W
    # Build a plane model (constant + x tilt + y tilt)
    A = xp.column_stack([
        xp.ones(valid.sum(), dtype=W.dtype),
        X[valid],
        Y[valid]
    ])
    # Fit the plane to the wavefront
    coeffs, _, _, _ = xp.linalg.lstsq(A, W[valid], rcond=None)
    piston, tilt_x, tilt_y = coeffs
    # Compute and subtract the fitted plane
    plane = piston + tilt_x * X + tilt_y * Y
    W_clean = W - plane
    # Keep invalid regions masked
    W_clean = xp.where(valid, W_clean, xp.nan)
    return W_clean

# This function computes basic wavefront quality metrics
def wavefront_metrics(W):
    # Convert input to NumPy or GPU array
    W = xp.asarray(W)
    # Consider only valid (finite) values
    valid = xp.isfinite(W)
    # Peak-to-valley: difference between highest and lowest points
    pv = xp.nanmax(W) - xp.nanmin(W)
    # RMS: overall strength of the wavefront error
    rms = xp.sqrt(xp.nanmean(W[valid] ** 2))
    return pv, rms

# This function visualizes the reconstructed wavefront in 3D
def plot_wavefront(X, Y, W, wavelength_nm=532):
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    # Force data to CPU for plotting
    X = xp.asnumpy(X)
    Y = xp.asnumpy(Y)
    W = xp.asnumpy(W)
    # Compute wavefront quality metrics
    pv, rms = wavefront_metrics(W)
    pv = float(pv)
    rms = float(rms)
    # Create a 3D surface plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, W, cmap="viridis", linewidth=0)
    # Display PV and RMS values in the title
    ax.set_title(
        f"Wavefront\nPV = {pv*1e9:.1f} nm | RMS = {rms*1e9:.1f} nm"
    )
    # Add a colorbar showing wavefront height in meters
    fig.colorbar(surf, shrink=0.6, label="Wavefront (m)")
    ax.set_xlabel("X (normalized)")
    ax.set_ylabel("Y (normalized)")
    ax.set_zlabel("W (meters)")
    plt.tight_layout()
    plt.show()

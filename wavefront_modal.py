# wavefront_modal.py
"""
Modal wavefront reconstruction from Shack–Hartmann slope data.

Inputs:
- Centroid positions (x, y)
- Local slopes (dx, dy)

Outputs:
- Zernike coefficients
- Reconstructed wavefront W(x, y)
- 3D surface plot (Matplotlib)
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper



def normalize_coordinates(centroids):
    """
    Normalize centroid coordinates to unit disk.
S
    Parameters
    ----------
    centroids : (N, 2) array
        Pixel coordinates of spot centroids.

    Returns
    -------
    x_norm, y_norm : arrays
        Normalized coordinates in [-1, 1]
    mask : boolean array
        True for points inside unit pupil
    """

    x = centroids[:, 0]
    y = centroids[:, 1]

    # Center the pupil
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Scale so max radius = 1
    r = np.sqrt(x**2 + y**2)
    r_max = np.max(r)

    x_norm = x / r_max
    y_norm = y / r_max

    mask = (x_norm**2 + y_norm**2) <= 1.0

    return x_norm[mask], y_norm[mask], mask
def compute_slopes(displacements, mask):
    """
    Extract slopes from displacement vectors.

    Parameters
    ----------
    displacements : (N, 2) array
        dx, dy from spot matching
    mask : boolean array
        Mask from pupil normalization

    Returns
    -------
    sx, sy : arrays
        Wavefront slopes
    """

    dx = displacements[:, 0][mask]
    dy = displacements[:, 1][mask]

    return dx, dy
def cartesian_to_polar(x, y):
    """
    Convert Cartesian coordinates to polar (rho, theta).

    x, y must already be normalized to unit disk.
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta
def zernike_radial(n, m, rho):
    """
    Radial part of Zernike polynomial.
    """
    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        num = (-1)**k * math.factorial(n - k)
        den = (
            math.factorial(k)
            * math.factorial((n + abs(m))//2 - k)
            * math.factorial((n - abs(m))//2 - k)
        )

        R += num / den * rho**(n - 2*k)
    return R
def zernike(n, m, rho, theta):
    """
    Zernike polynomial Z_n^m
    """
    R = zernike_radial(n, m, rho)
    if m > 0:
        return R * np.cos(m * theta)
    elif m < 0:
        return R * np.sin(-m * theta)
    else:
        return R
def numerical_derivative(func, x, y, eps=1e-6):
    """
    Compute numerical derivatives d/dx and d/dy.
    """
    f_x1 = func(x + eps, y)
    f_x2 = func(x - eps, y)
    dfdx = (f_x1 - f_x2) / (2 * eps)

    f_y1 = func(x, y + eps)
    f_y2 = func(x, y - eps)
    dfdy = (f_y1 - f_y2) / (2 * eps)

    return dfdx, dfdy
def zernike_derivatives(n, m, x, y):
    """
    Compute dZ/dx and dZ/dy for Zernike Z_n^m
    """
    def Z(xi, yi):
        rho, theta = cartesian_to_polar(xi, yi)
        return zernike(n, m, rho, theta)

    return numerical_derivative(Z, x, y)
def generate_zernike_modes(n_modes, include_piston=False):
    modes = []

    if include_piston:
        modes.append((0, 0))

    n = 1
    while len(modes) < n_modes:
        for m in range(-n, n + 1, 2):
            if len(modes) >= n_modes:
                break
            modes.append((n, m))
        n += 1

    return modes

def normalize_centroids(centroids):
    centroids = np.asarray(centroids)

    cx = np.mean(centroids[:, 0])
    cy = np.mean(centroids[:, 1])

    x = centroids[:, 0] - cx
    y = centroids[:, 1] - cy

    r = np.sqrt(x**2 + y**2)
    R_px = np.max(r)

    x_norm = x / R_px
    y_norm = y / R_px

    return x_norm, y_norm, R_px

def build_design_matrix(x, y, modes, R_phys):
    N = len(x)
    M = len(modes)
    A = np.zeros((2*N, M))

    for k in range(N):
        for j, (n, m) in enumerate(modes):
            dZdx, dZdy = zernike_derivatives(n, m, x[k], y[k])

            # Physical scaling (CRITICAL)
            A[2*k, j]     = dZdx / R_phys
            A[2*k + 1, j] = dZdy / R_phys

    return A


def build_slope_vector(displacements):
    """
    Stack x- and y-slopes into vector b.

    displacements: (N, 2)

    Returns:
        b: shape (2N,)
    """
    N = displacements.shape[0]
    b = np.zeros(2 * N)

    for k in range(N):
        b[2*k]     = displacements[k, 0]
        b[2*k + 1] = displacements[k, 1]

    return b
def solve_modal_coefficients(A, b):
    """
    Solve least-squares problem for modal coefficients.

    Returns:
        C: Zernike coefficients
    """
    C, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return C
def is_valid_zernike(n, m):
    return (
        n >= 0 and
        abs(m) <= n and
        (n - abs(m)) % 2 == 0
    )

def reconstruct_wavefront(coeffs, modes, grid_size=200):
    # -------------------------------
    # Zernike mode validity check
    # -------------------------------
    def is_valid_zernike(n, m):
        return (
            n >= 0 and
            abs(m) <= n and
            (n - abs(m)) % 2 == 0
        )

    valid = [
        (c, (n, m))
        for c, (n, m) in zip(coeffs, modes)
        if np.isfinite(c) and is_valid_zernike(n, m)
    ]

    if not valid:
        raise ValueError("No valid Zernike modes passed to reconstruct_wavefront()")

    coeffs, modes = zip(*valid)

    # -------------------------------
    # Normalized reconstruction grid
    # -------------------------------
    x = np.linspace(-1.0, 1.0, grid_size)
    y = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(x, y, indexing="xy")

    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # -------------------------------
    # Pupil mask
    # -------------------------------
    pupil = rho <= 1.0

    # -------------------------------
    # Wavefront reconstruction
    # -------------------------------
    W = np.zeros_like(X, dtype=np.float64)

    for c, (n, m) in zip(coeffs, modes):
        Z = np.zeros_like(W)
        Z[pupil] = zernike(n, m, rho[pupil], theta[pupil])
        W += c * Z

    # -------------------------------
    # Mask outside pupil AFTER sum
    # -------------------------------
    W[~pupil] = np.nan

    return X, Y, W

def remove_piston_tilt(W, X, Y):
    """
    Remove piston and tilt from reconstructed wavefront.

    Fits: W = a + bX + cY over the valid pupil and subtracts it.
    """

    valid = np.isfinite(W)

    if np.sum(valid) < 3:
        return W  # not enough points to fit

    A = np.column_stack([
        np.ones(valid.sum()),
        X[valid],
        Y[valid]
    ])

    coeffs, _, _, _ = np.linalg.lstsq(A, W[valid], rcond=None)
    piston, tilt_x, tilt_y = coeffs

    plane = piston + tilt_x * X + tilt_y * Y
    W_clean = W - plane

    # 🔑 Preserve NaNs outside pupil
    W_clean[~valid] = np.nan

    return W_clean

def wavefront_metrics(W):
    valid = ~np.isnan(W)
    pv = np.nanmax(W) - np.nanmin(W)
    rms = np.sqrt(np.nanmean(W[valid]**2))
    return pv, rms
def plot_wavefront(X, Y, W, wavelength_nm=532):
    from mpl_toolkits.mplot3d import Axes3D

    pv, rms = wavefront_metrics(W)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, W, cmap="viridis", linewidth=0)

    ax.set_title(
        f"Wavefront\nPV = {pv*1e9:.1f} nm | RMS = {rms*1e9:.1f} nm"
    )

    fig.colorbar(surf, shrink=0.6, label="Wavefront (m)")

    ax.set_xlabel("X (normalized)")
    ax.set_ylabel("Y (normalized)")
    ax.set_zlabel("W (meters)")

    plt.tight_layout()
    plt.show()

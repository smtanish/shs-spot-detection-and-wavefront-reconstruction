
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from backend import xp, GPU_ENABLED

_ZERNIKE_GRID_CACHE = {}

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper
def normalize_coordinates(centroids):

    centroids = xp.asarray(centroids)

    x = centroids[:, 0]
    y = centroids[:, 1]

    # Center the pupil
    x = x - xp.mean(x)
    y = y - xp.mean(y)

    # Scale so max radius = 1
    r = xp.sqrt(x**2 + y**2)
    r_max = xp.max(r)

    x_norm = x / r_max
    y_norm = y / r_max

    mask = (x_norm**2 + y_norm**2) <= 1.0

    return x_norm[mask], y_norm[mask], mask

def compute_slopes(displacements, mask):

    displacements = xp.asarray(displacements)

    dx = displacements[:, 0][mask]
    dy = displacements[:, 1][mask]

    return dx, dy
def cartesian_to_polar(x, y):
    x = xp.asarray(x)
    y = xp.asarray(y)

    rho = xp.sqrt(x**2 + y**2)
    theta = xp.arctan2(y, x)

    return rho, theta

def zernike_radial(n, m, rho):
    rho = xp.asarray(rho)
    R = xp.zeros_like(rho)

    for k in range((n - abs(m)) // 2 + 1):
        num = (-1)**k * math.factorial(n - k)
        den = (
            math.factorial(k)
            * math.factorial((n + abs(m)) // 2 - k)
            * math.factorial((n - abs(m)) // 2 - k)
        )

        R = R + (num / den) * rho ** (n - 2*k)

    return R

def zernike(n, m, rho, theta):
    rho = xp.asarray(rho)
    theta = xp.asarray(theta)

    R = zernike_radial(n, m, rho)

    if m > 0:
        return R * xp.cos(m * theta)
    elif m < 0:
        return R * xp.sin(-m * theta)
    else:
        return R

def numerical_derivative(func, x, y, eps=1e-6):
    x = xp.asarray(x)
    y = xp.asarray(y)

    eps_xp = xp.asarray(eps)

    f_x1 = func(x + eps_xp, y)
    f_x2 = func(x - eps_xp, y)
    dfdx = (f_x1 - f_x2) / (2 * eps_xp)

    f_y1 = func(x, y + eps_xp)
    f_y2 = func(x, y - eps_xp)
    dfdy = (f_y1 - f_y2) / (2 * eps_xp)

    return dfdx, dfdy

def zernike_derivatives(n, m, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    def Z(xi, yi):
        rho, theta = cartesian_to_polar(xi, yi)
        return zernike(n, m, rho, theta)

    dZdx, dZdy = numerical_derivative(Z, x, y)
    return dZdx, dZdy

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
    centroids = xp.asarray(centroids)

    cx = xp.mean(centroids[:, 0])
    cy = xp.mean(centroids[:, 1])

    x = centroids[:, 0] - cx
    y = centroids[:, 1] - cy

    r = xp.sqrt(x**2 + y**2)
    R_px = xp.max(r)

    x_norm = x / R_px
    y_norm = y / R_px

    return x_norm, y_norm, R_px
def build_design_matrix(x, y, modes, R_phys):
    # Force CPU NumPy arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    R_phys = float(R_phys)

    N = x.shape[0]
    M = len(modes)

    A = np.zeros((2 * N, M), dtype=np.float64)

    for j, (n, m) in enumerate(modes):
        dZdx, dZdy = zernike_derivatives(n, m, x, y)

        A[0::2, j] = dZdx / R_phys
        A[1::2, j] = dZdy / R_phys

    return A

def build_slope_vector(displacements):

    displacements = xp.asarray(displacements)

    N = displacements.shape[0]
    b = xp.zeros(2 * N, dtype=displacements.dtype)

    b[0::2] = displacements[:, 0]
    b[1::2] = displacements[:, 1]

    return b

def solve_modal_coefficients(A, b):

    A = xp.asarray(A)
    b = xp.asarray(b)

    C, residuals, rank, s = xp.linalg.lstsq(A, b, rcond=None)
    return C
def is_valid_zernike(n, m):
    return (
        n >= 0 and
        abs(m) <= n and
        (n - abs(m)) % 2 == 0
    )
def reconstruct_wavefront(coeffs, modes, grid_size=200):
    coeffs = np.asarray(coeffs, dtype=np.float64)

    # ---- Filter valid modes ----
    valid = [
        (c, (n, m))
        for c, (n, m) in zip(coeffs, modes)
        if np.isfinite(c) and is_valid_zernike(n, m)
    ]

    if not valid:
        raise ValueError("No valid Zernike modes passed to reconstruct_wavefront()")

    coeffs, modes = zip(*valid)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    modes = tuple(modes)  # needed for cache key

    M = len(coeffs)

    # ---- Cache key ----
    cache_key = (grid_size, modes)

    if cache_key in _ZERNIKE_GRID_CACHE:
        X, Y, pupil, Z_stack = _ZERNIKE_GRID_CACHE[cache_key]
    else:
        # ---- Build grid ----
        x = np.linspace(-1.0, 1.0, grid_size)
        y = np.linspace(-1.0, 1.0, grid_size)
        X, Y = np.meshgrid(x, y, indexing="xy")

        rho = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        pupil = rho <= 1.0

        # ---- Precompute all Zernike modes ----
        Z_stack = np.zeros((M, grid_size, grid_size), dtype=np.float64)

        for j, (n, m) in enumerate(modes):
            Zj = np.zeros_like(rho)
            Zj[pupil] = zernike(n, m, rho[pupil], theta[pupil])
            Z_stack[j] = Zj

        _ZERNIKE_GRID_CACHE[cache_key] = (X, Y, pupil, Z_stack)

    # ---- Weighted sum ----
    W = np.tensordot(coeffs, Z_stack, axes=(0, 0))
    W[~pupil] = np.nan

    return X, Y, W

def remove_piston_tilt(W, X, Y):
    W = xp.asarray(W)
    X = xp.asarray(X)
    Y = xp.asarray(Y)

    valid = xp.isfinite(W)

    if xp.sum(valid) < 3:
        return W  

    A = xp.column_stack([
        xp.ones(valid.sum(), dtype=W.dtype),
        X[valid],
        Y[valid]
    ])

    coeffs, _, _, _ = xp.linalg.lstsq(A, W[valid], rcond=None)
    piston, tilt_x, tilt_y = coeffs

    plane = piston + tilt_x * X + tilt_y * Y
    W_clean = W - plane

    W_clean = xp.where(valid, W_clean, xp.nan)

    return W_clean
def wavefront_metrics(W):
    W = xp.asarray(W)

    valid = xp.isfinite(W)
    pv = xp.nanmax(W) - xp.nanmin(W)
    rms = xp.sqrt(xp.nanmean(W[valid] ** 2))

    return pv, rms

def plot_wavefront(X, Y, W, wavelength_nm=532):
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # ---- Force CPU for plotting ----
    X = xp.asnumpy(X)
    Y = xp.asnumpy(Y)
    W = xp.asnumpy(W)

    pv, rms = wavefront_metrics(W)
    pv = float(pv)
    rms = float(rms)

    fig = plt.figure(figsize=(8, 6))
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

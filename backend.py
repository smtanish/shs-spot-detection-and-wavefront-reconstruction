
GPU_ENABLED = False

try:
    import cupy as cp
    xp = cp
    GPU_ENABLED = True
except Exception:
    import numpy as np
    xp = np
    GPU_ENABLED = False

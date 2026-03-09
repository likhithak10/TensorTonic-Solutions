import numpy as np
import math
import scipy.special as sp

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # issues with ReLU (0, max) -> head neurons (gradient always is 0), non-smooth gradients (not stable)
    # GELU -> smooth transtion  

    x = np.array(x)
    res = (1/2) * (x) * (1 + sp.erf((x / 2 ** 0.5)))
    return res
    
    pass

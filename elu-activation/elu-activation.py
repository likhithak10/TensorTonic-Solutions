import numpy as np

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # ReLU -> leads to non-linearity, passes positve inputs and outputs negative inputs as 0
    # ELU leads to a smooth curve (unlike ReLU), ELU has a non-zero, smooth slope for negative values

    x = np.asanyarray(x)
    res = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return res.tolist()
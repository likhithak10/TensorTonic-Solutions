import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # creates smoother gradient, sigmoinf is from 0 to 1 noninclusive and we multiply that by x, smoother slope to negative values

    x = np.array(x)
    return (x) * ((1) / (1 + np.exp(-x)))
    pass
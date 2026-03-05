import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # The sigmoid fucntion condenses any real number into the range of [0, 1]
    x = np.array(x)
    return 1 / (1 + np.exp(-x))
    pass
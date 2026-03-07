import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # this function is zero centered unlike the sigmoid function
    x = np.array(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    pass
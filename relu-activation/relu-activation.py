import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # ReLU leads to nonlinerity -> putputs + as + and - as 0

    x = np.array(x)
    return np.maximum(0,x)
    pass
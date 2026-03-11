import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # modifies negative values to have small slope instead of 0

    x = np.asanyarray(x)
    return np.where(x>=0, x, (alpha) * x)
    
    pass
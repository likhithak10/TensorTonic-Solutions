import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    euc_a = np.linalg.norm(a) 
    euc_b = np.linalg.norm(b) 

    if euc_a == 0 or euc_b == 0:
        return 0
        
    return (np.dot(a, b)) / (euc_a * euc_b)
    pass
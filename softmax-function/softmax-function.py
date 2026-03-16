import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # turns numbers (negative inclusive) into probabilites as their class numbers can be hard to understand for a classification model
    # converts the values between 0 and 1 -> adding the values of all the classes in the output of the softmax will sum to 1
    # for 2d array, treat each row as a seperate array od data (set axis=1)

    # didn't read the whole description beforr seeing the numerical overview problrm and how to deal with it but stack overflow helped. going to work on numpy operations more and refactor the code to be optimal

    x = np.array(x)

    if len(x.shape) > 1:
        total_sum = np.sum(np.exp(x), axis=1)
        return np.exp(x) / total_sum[:, np.newaxis]
    else:
        total_sum = np.sum(np.exp(x - np.max(x)))
        return np.exp(x - np.max(x)) / total_sum
    
    pass
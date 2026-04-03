import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)

    return (1/len(y_pred)) * (np.sum(np.pow(y_pred - y_true, 2)))
    pass

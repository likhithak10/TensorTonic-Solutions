import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # hinge loss primarlity supports svms (classification hyperplanes) which work on classification tasks -> it forces model to predict with high confidence 
    # loss is 0 for correct + confident predictions else (linearly increases)
    
    y_score, y_true = np.array(y_score), np.array(y_true)
    res = np.maximum(0, margin - (y_score*y_true))
    if reduction == "mean":
        return np.mean(res)
    else:
        return np.sum(res)
    pass
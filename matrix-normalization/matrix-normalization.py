import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    matrix = np.array(matrix)
    if matrix.ndim != 2:
        return None
        
    if axis not in (0, 1, None):
        return None

    seen = ('l1', 'l2', 'max')
    
    if norm_type not in seen:
        return None

    if norm_type == 'l1':
        denom = np.sum(np.abs(matrix), axis=axis, keepdims=True)

    elif norm_type == 'l2':
        if not axis:
            denom = np.linalg.norm(matrix, ord=None, axis=axis, keepdims=True)
        else:
            denom = np.linalg.norm(matrix, ord=2, axis=axis, keepdims=True)

    elif norm_type == 'max':
        denom = np.max(np.abs(matrix), axis=axis, keepdims=True)

    return matrix / np.maximum(denom, 1e-12)
    
    pass

    # norm = {
    #     'l1' : 1,
    #     'l2' : 2,
    #     'max' : np.inf
    # }

    # norm = np.linalg.norm(matrix, ord=norm[norm_type], axis=axis, keepdims=True)

    # return matrix / np.maximum(norm, 1e-12)
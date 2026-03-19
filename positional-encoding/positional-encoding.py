import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # positional encoding allows there to be position information into the model by adding posiiton dependant vectors -> example: the word order of a sentence matters

    ## WITH VECTORIZATION
    # create [X] by [1] matrix
    pos = np.arange(seq_len)[:, np.newaxis]

    #np.arange(start, stop, step)
    i = np.arange(0, d_model, 2)
    denomintaor = np.power(base,(i)/d_model)

    res = np.zeros((seq_len, d_model))

    res[:, 0::2] = np.sin(pos / denomintaor)
    #need to slice odd denom incase last column
    res[:, 1::2] = np.cos(pos / denomintaor[:d_model // 2])
    return res
    pass
    
    

    ## WITH LOOP 
    # res = np.zeros((seq_len, d_model))

    # for x in range(seq_len):
    #     for i in np.arange(int(d_model/2)):
    #         res[x, 2*i] = np.sin((x) / (np.power(base, 2*i/d_model))) 
    #         res[x, 2*i+1] = np.cos((x) / (np.power(base, 2*i/d_model))) 
    # return res
    # pass
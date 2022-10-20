# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns optimal weights and mse.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar denoting the loss computed as MSE

    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    return (w := np.linalg.solve(a,b)), compute_mse(y, tx, w)

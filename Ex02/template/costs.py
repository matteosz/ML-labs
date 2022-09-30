# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np

class InvalidLossFunction(Exception):
    print("The loss function inserted hasn't been implemented")

def mse(e): # Mean Squared Error
    return .5 * np.mean(e**2)

def mae(e): # Mean Absolute Error
    return np.mean(np.abs(e))

def compute_loss(y, tx, w, mod='MSE'):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.
        mod: string, default='MSE'

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.dot(tx, w)
    
    if mod == 'MSE': 
        return mse(e)
    
    if mod == 'MAE': 
        return mae(e)
    
    raise InvalidLossFunction
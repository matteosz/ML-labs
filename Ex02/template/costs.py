# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


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
        return 0.5/len(y) * np.dot(e.T, e)
    
    if mod == 'MAE': 
        return 1/len(y) * np.sum(np.abs(e), axis=0)
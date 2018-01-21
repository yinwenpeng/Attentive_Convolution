# -*- coding: utf-8 -*-
import numpy as np


def softmax(M):
    """Calculate the row-wise softmax given a matrix.

    Parameters
    ----------
    M : 2d structure (m x n)

    Returns
    -------
    ndarray(m x n)
        probabilities according to softmax computation, each row sum = 1
    """
    M = np.asarray(M)

    if M.ndim == 1:
        M = np.atleast_2d(M)

    maxes = np.amax(M, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(M - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True)
    return dist

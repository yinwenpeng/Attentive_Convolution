# -*- coding: utf-8 -*-
import numpy as np


def interpolate(model1, model2, weight):
    """Interpolate the probabilities of two models.

    Model 1 is weighted by the parameter, model 2 is weighted by (1 - weight).

    Parameters
    ----------
    model1 : ndarray
        probabilities of model 1
    model2 : ndarray
        probabilities of model 2
    weight : float
        weight of model 1, model 2 will receive weight (1 - weight)

    Returns
    -------
    float
        interpolated probability
    """
    model1 = np.asarray(model1)
    model2 = np.asarray(model2)
    interpolated = weight * model1 + (1-weight) * model2
    return perplexity(interpolated)

def perplexity(probabs):
    """Calculate perplexity given the list of probabs.

    Parameters
    ----------
    probabs : iterable
        list of probabs

    Returns
    -------
    float
        perplexity
    """
    probabs = np.asarray(probabs)
    return np.exp(-np.sum(np.log(probabs)) / np.max(probabs.shape))


# -*- coding: utf-8 -*-
"""
This file contains common utility classes and methods for classification.
"""
from sklearn.metrics.metrics import accuracy_score, \
    precision_recall_fscore_support


def calc_metrics(true_labels, predicted_labels):
    """Provide accuracy, precision, recall, and f1 as error measure.

    Parameters
    ----------
    true_labels : list, ndarray
        true labels
    predicted_labels : list, ndarray
        predicted labels

    Returns
    -------
    (float, float, float, float)
        accuracy, precision, recall, f1

    Example
    -------
    >>> y_true = [0, 1, 1, 0]
    >>> y_pred = [0, 0, 1, 1]
    >>> calc_metrics(y_true, y_pred)
    (0.5, 0.5, 0.5, 0.5)
    """
    acc = accuracy_score(true_labels, predicted_labels)
    p, r, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels,
            average='micro')
    return (acc, p, r, f1)

# -*- coding: utf-8 -*-
import numpy as np

from cis.deep.utils import utf8_file_open, file_line_generator


def calc_matrix_statistics(matrix_file):
    """Calculates some basic statistics for huge matrix files.

    If a matrix is too big to be imported into a program, use this method to
    calculate the mean, maximum, minimum, and standard deviation of every line
    in the file. It returns a generator.

    Parameters
    ----------
    matrix_file : str
        filename of the matrix file; the file must be a csv file with spaces as
        separator

    Returns
    -------
    generator : (float, float, float, float)
        mean, max, min, std_dev of current line in the matrix file
    """

    for line in file_line_generator(matrix_file):
        a = np.fromstring(line, sep=u' ')
        yield (np.mean(a), np.max(a), np.min(a), np.std(a))

    raise StopIteration()

def cross_entropy(p, q):
    """Calculate the cross-entropy for discrete probability distributions.

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.

    Returns
    -------
    cross entropy
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return -np.sum(p * np.log10(q))

def kullback_leibler(p, q):
    """Calculate the Kullback-Leibler divergence D(P || Q) for discrete
    probability distributions.

    Source taken from https://gist.github.com/larsmans/3104581
    The computation does not consider those elements i where either p_i or q_i
    is zero.

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.

    Returns
    -------
    KL divergence
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(np.logical_and(p != 0, q != 0), p * np.log(p / q), 0))

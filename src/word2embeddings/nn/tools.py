#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tools.py: Tools for dealing with Networks."""

from cis.deep.utils import file_line_generator
import numpy as np


LOG_FORMAT = '%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s'


def read_unigram_distribution(filename):
    """Read the unigram distribution for all vocabulary items from the file.

    1 probability per line.
    Caution: Don't forget to add the 4 special tokens, e.g., <UNK>. Besides
    <UNK> we don't want to draw them as noise, therefore they should have
    a count of 0.
    """
    unigram_dist = read_unigram_frequencies(filename)

    # Note: use the same datatype as Theano's floatX here, to avoid problems.
    return np.asarray(unigram_dist, 'float32') / np.sum(unigram_dist)

def read_unigram_frequencies(filename):
    """Read the unigram frequencies for all vocabulary items from the file.

    1 frequency per line.
    Caution: Don't forget to add the 4 special tokens, e.g., <UNK>. Besides
    <UNK> we don't want to draw them as noise, therefore they should have
    a count of 0.
    """
    unigram_dist = []

    for line in file_line_generator(filename):
        unigram_dist.append(int(line))

    return unigram_dist

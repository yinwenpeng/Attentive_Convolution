#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""util.py: Useful functions"""

import numpy
from theano import tensor as T


def zero_value(shape, dtype):
    return numpy.zeros(shape, dtype=dtype)

def random_value_GloBen10(shape, dtype, random_generator=None, no_of_units=None):
    """
    Return a randomly initialized matrix using a uniform distribution.

    Returns a randomly initialized matrix using the method proposed in
    [GloBen10].

    Parameters
    ----------
    shape : (int, int)
        size of the matrix that needs to be initialized
    dtype : dtype
        datatype of the random values
    random_generator : numpy.random.RandomState
        random number generator; if None a new instance will automatically be
        created
    no_of_units : (int, int)
        number of input and output dimensions; if None it will be the same as
        shape
    """
    # `W` is initialized with `W_values` which is uniformely sampled
    # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
    # for tanh activation function
    # the output of uniform if converted using asarray to dtype
    # theano.config.floatX so that the code is runable on GPU
    # Note : optimal initialization of weights is dependent on the
    #                activation function used (among other things).
    #                For example, results presented in [Xavier10] suggest that you
    #                should use 4 times larger initial weights for sigmoid
    #                compared to tanh
    #                We have no info for other function, so we use the same as
    #                tanh.
    if not random_generator:
        random_generator = numpy.random.RandomState(1234)

    if no_of_units is None:
        total_dimensions = numpy.sum(shape)
    else:
        total_dimensions = numpy.sum(no_of_units)

    low = -numpy.sqrt(6. / total_dimensions)
    high = numpy.sqrt(6. / total_dimensions)
    random_values = random_generator.uniform(low=low, high=high, size=shape)
    W_values = numpy.asarray(random_values, dtype=dtype)
    return W_values

def random_value_normal(shape, dtype, random_generator=None):
    """Return a randomly initialized matrix using a normal distribution.

    Returns random numbers from a zero-mean Gaussian with 0.01 std dev. This
    std dev value has been proposed by [Hin10].

    Parameters
    ----------
    shape : (int, int)
        size of the matrix that needs to be initialized
    dtype : dtype
        datatype of the random values
    random_generator : numpy.random.RandomState
        random number generator; if None a new instance will automatically be
        created
    """

    if not random_generator:
        random_generator = numpy.random.RandomState(1234)

    random_values = random_generator.normal(scale=0.01, size=shape)
    W_values = numpy.asarray(random_values, dtype=dtype)
    return W_values

def threshold(x):
    """An approximation of sigmoid.

    More approximate and faster than ultra_fast_sigmoid.

    Approx in 3 parts: 0, scaled linear, 1

    Removing the slope and shift does not make it faster.

    """
#     x = theano.printing.Print('x')(x)
#     gt = theano.printing.Print('gt')(T.gt(x, 0.5))
#     return gt
    return T.gt(x, 0.5)


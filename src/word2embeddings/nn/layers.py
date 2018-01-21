#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Construct fully connected layers on top of theano model interface."""

from collections import OrderedDict
from logging import getLogger
import os

from numpy import ndarray
import scipy.io
from theano import config
from theano import tensor as T
import theano
from theano.compile.function_module import Function, FunctionMaker
from theano.ifelse import ifelse
from theano.sandbox.cuda.var import CudaNdarrayVariable, \
    CudaNdarraySharedVariable
import theano.sparse
from theano.sparse.basic import SparseVariable
from theano.tensor.basic import TensorVariable
from theano.tensor.nnet.nnet import binary_crossentropy
from theano.tensor.sharedvar import TensorSharedVariable, ScalarSharedVariable

from cis.deep.utils import logger_config
import numpy as np
from word2embeddings.nn.util import random_value_GloBen10, zero_value


floatX = config.floatX
intX = 'int64'

log = getLogger(__name__)
logger_config(log)

class Error(Exception):
    """Base class for module specific Errors."""


class PathError(Error):
    """ Raised if expected files are not found."""


class MissingInfoError(Error, ValueError):
    """ Raised if the network is missing a key component."""


class SerializationLayer(object):
    """ Implements safe serialization criterion for Theano graphs.

    Caution: All ndarrays will be converted into shared variables. I.e., if
    there's a normal ndarray in the class, it will converted, too. You can
    prevent that by adding their name into the keep_as_ndarray list.
    """
    # Indicates which variables should not be converted into a Theano shared
    # variables even though they are ndarrays
    keep_as_ndarray = []

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def link(self, inputs):
        """ Defines the relation between inputs and the output of this layer.
        Basically, this method constructs the graph of the computation on top of
        the parameters that should be defined in __init__. This method will be
        called after unpickling.
        """
        raise NotImplementedError

    def build(self):
        """ Builds the output function of this layer.
        This will be called after unpickling the model.
        """
        self.forward_pass = theano.function(inputs=self.inputs,
                outputs=self.outputs)

    def load_params(self, base_filename, params_str):
        """Load the listed model parameters from separate files.

        All parameters, that are mentioned in params_str are loaded from
        separate files that are named "base_filename.param_name".
        Caution: Remember to call the link method after loading parameters form
        files.

        Parameters
        ----------
        base_filename : str
            basic filename of input parameters
        params_str : str
            comma separated list of parameter names
        """

        for p in params_str.split(','):
            filename = base_filename + u'.' + p

            if not os.path.exists(filename):

                if os.path.exists(filename + u'.gz'):
                    filename += u'.gz'
                elif os.path.exists(filename + u'.bz2'):
                    filename += u'.bz2'
                elif os.path.exists(filename + u'.mtx'):
                    filename += u'.mtx'
                else:
                    raise IOError('parameter file for parameter %s ' +
                            'at location %s not found' % (p, filename))

            if filename.endswith(".mtx"):
                value = scipy.io.mmread(filename)
                # Currently Theano supports only CSR or CSC forman, so convert COO to CSR
                value = value.tocsr()
            else:
                # Caution: We need to convert the filename into a string (not
                # unicode), otherwise loading a .gz file doesn't work. That's a bug
                # in numpy.
                value = np.loadtxt(str(filename), dtype=floatX)

            if p in self.keep_as_ndarray:
                self.__dict__[p] = value
            else:
                self.__dict__[p] = theano.shared(value=value, name=p)

    def store_params(self, base_filename, params_str, compress=False,
            fmt='txt'):
        """Store listed model parameters in separate files.

        All parameters, that are mentioned in params_str are stored in separate
        files named "base_filename.param_name".

        Parameters
        ----------
        base_filename : str
            basic filename of the target output
        params_str : str
            comma separated list of parameter names
        compress : bool
            indicate whether or not to compress the output using gzip
        fmt : str
            indicates the format to store the parameters in
            txt = space separated csv format (slow, open format)
            npy = binary numpy format (faster, but proprietary)
            mtx = MatrixMarket format (for saving sparse matrices)
        """
        file_extension = u'.gz' if compress else u''

        if fmt == 'txt':
            save_method = np.savetxt
        elif fmt == 'npy':
            save_method = np.save
        elif fmt == 'mtx':
            save_method = scipy.io.mmwrite
        else:
            raise ValueError('unknown format to store parameters in')

        for p in params_str.split(','):
            log.info('storing parameter %s' % p)

            filename = base_filename + u'.' + p + file_extension
            values = self.__dict__[p]

            if p not in self.keep_as_ndarray:
                values = values.get_value()

            if fmt == 'mtx':
                # save matrix as COO, not CSR
                values = values.tocoo()

            save_method(filename, values)


class Layer(SerializationLayer):
    """ General abstraction for a neural network layer."""

    def __init__(self, name):
        super(Layer, self).__init__()
        self.name = name
        self._params = []  # these parameters get updated automatically
        self.lr = theano.shared(value=np.cast[floatX](1.0),
                name='weights_lr_' + self.name)

    def params(self):
        """ Iterates over the parameters that have to be learned."""
        return []

    def update_learning_rate(self, remaining):
        """Update the learning rate depending on a given method.

        Three methods are available: fan_in, linear, global (no change).
        """

        if not self.params():
            return

        new_value = self.global_lr

        if self.lr_adaptation_method == 'linear':
            new_value = self.global_lr * remaining

        if self.lr_method == 'fan_in':
            new_value /= self.fan_in

        self.lr.set_value(new_value)

        log.debug('Param {} \'s learning rate is {:e}'.format(self.name,
                new_value))

    def set_learning_rate(self, lr, method='global', lr_adaptation_method=''):
        """Set layer global lr and update the local lr of the layers"""
        self.lr_method = method
        self.global_lr = lr
        self.lr_adaptation_method = lr_adaptation_method
        self.update_learning_rate(1.0)

    def updates(self, cost):
        """ Layer required updates for each training batch.

        Return  the variables and their update functions for every internal
        parameter.

        Parameters
        ----------
        cost : ?
            cost function
        """
        learn_rate_val = np.asscalar(self.lr.get_value())

        for param in self.params():
            log.debug('Param {} \'s learning rate is {:e}'.format(param.name,
                    learn_rate_val))

            if self.clipping:
                grad_ = T.grad(cost=cost, wrt=param)
                grad_norm = T.sum(grad_ * grad_) ** 0.5
                scaled = grad_ * self.threshold / grad_norm
                grad = ifelse(T.gt(grad_norm, self.threshold), scaled, grad_)
                yield (param, param - self.lr * grad)
            else:
                yield (param, param - self.lr * T.grad(cost=cost, wrt=param))

    def status(self):
        for param in self.params():
            min_value = param.min().eval()
            max_value = param.max().eval()
            avg_value = param.mean().eval()
            stats = 'min:%f\taverage:%f\tmax:%f' % (min_value, avg_value,
                    max_value)
            yield (param, stats)


class HiddenLayer(Layer):
    """ A fully connected layer that holds parameters that have to be learned.
    """

    def __init__(self, name='HiddenLayer', shape=(0, 0), w_values=None,
            activation=T.tanh):

        super(HiddenLayer, self).__init__(name)
        self.activation = activation

        if w_values is None:
            input_dim, output_dim, = shape
            w_values = random_value_GloBen10((input_dim, output_dim), floatX)
        if activation == theano.tensor.nnet.sigmoid:
            w_values *= 4

        self.fan_in = w_values.shape[0]
        self.clipping = False
        self.threshold = 1e12
        self.weights = theano.shared(value=w_values, name='weights_' + self.name)

    def params(self):
        yield self.weights

    def size(self):
        """ Returns the number of the parameters without considering the bias.
        """
        return self.weights.size

    def link(self, inputs):
        """ Once the inputs are determined we construct the output function.
                output = inputs[0] * Weights

        Parameters
        ----------
        inputs
            0) input values for this hidden layer

        Returns
        -------
        []
            0) non-linear output of the hidden layer (after activation function)
            1) linear output of the hidden layer (before activation function)
        """
        self.L1 = T.sum(T.abs_(self.weights))
        self.L2 = T.sum((self.weights ** 2))

        self.inputs = inputs
        self.linear_output = T.dot(self.inputs[0], self.weights)

        # No activation function given -> linear model
        if not self.activation:
            self.nonlinear_output = self.linear_output
        else:
            self.nonlinear_output = self.activation(self.linear_output)

        self.outputs = [self.nonlinear_output, self.linear_output]
        return self.outputs


class BiasedHiddenLayer(HiddenLayer):
    """ Hidden layer without biased activations."""

    def __init__(self, b_values=None, **kwargs):
        super(BiasedHiddenLayer, self).__init__(**kwargs)

        if b_values is None:
            output_dim = self.weights.eval().shape[1]
            b_values = zero_value((output_dim,), type=floatX)

        self.bias = theano.shared(value=b_values, name='bias_' + self.name)

    def params(self):
        yield self.weights
        yield self.bias

    def link(self, inputs):
        """ Once the inputs are determined we construct the output function.
                output = inputs[0] * Weights + bias

        Parameters
        ----------
        inputs
            0) input values for this hidden layer

        Returns
        -------
        []
            0) non-linear output of the hidden layer (after activation function)
            1) linear output of the hidden layer (before activation function)
        """
        _, linear_out = super(BiasedHiddenLayer, self).link(inputs)
        self.linear_output = linear_out + self.bias

        # No activation function given -> linear model
        if not self.activation:
            self.nonlinear_output = self.linear_output
        else:
            self.nonlinear_output = self.activation(self.linear_output)

        self.outputs = [self.nonlinear_output, self.linear_output]
        return self.outputs


class EmbeddingLayer(HiddenLayer):
    """
    EmbeddingsLayer is a layer where a lookup operation is performed.
    Indices supplied as input replaced with their embedding representation.
    This is done using Theano operators that support backpropagation.

    The embeddings are stored in matrix 'Weights'.
    """

    def __init__(self, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.fan_in = 1.0

    def link(self, inputs):
        """ Input should be a matrix with the rows representing examples.

        We need embeddings for all indices in a matrix, that's why we need to
        flatten the matrix first, get all embeddings for the indices and than
        reshape it again.

        Parameters
        ----------
        inputs
            0) indexes of the examples to get the embeddings of

        Returns
        -------
        []
            0) embeddings of the input
        """
        self.inputs = inputs
        input = self.inputs[0]
        concatenated_input = input.flatten()

        # Rami's fix
        if config.device == 'gpu':
            indexed_rows = self.weights[concatenated_input]
        else:
            indexed_rows = theano.sparse_grad(self.weights[concatenated_input])

        concatenated_rows = indexed_rows.flatten()
        num_examples = input.shape[0]
        width = concatenated_rows.size // num_examples
        self.outputs = [concatenated_rows.reshape((num_examples, width))]
        return self.outputs


class LossLayer(Layer):
    """ A layer without parameters."""

    def params(self):
        return []


class SquaredErrorLossLayer(LossLayer):
    """Computes the euclidean distance between two vectors.

    squared error: ||y - activation||^2, where y is the target vector
    """

    def __init__(self, name='SquaredErrorLossLayer'):
        super(SquaredErrorLossLayer, self).__init__(name=name)

    def link(self, inputs):
        """

        Parameters
        ----------
        inputs
            0) predicted network output
            1) correct output

        Returns
        -------
        []
            0) see cost()
            1) see errors()
        """
        self.inputs = inputs
        self.predicted_output = self.inputs[0]
        self.correct_output = self.inputs[1]
        self.outputs = [self.cost(), self.errors()]
        return self.outputs

    def cost(self):
        """Calculate the average cost per example."""
#         diff = theano.printing.Print('difference')(self.correct_output - self.predicted_output)
#         sqr = theano.printing.Print('square')(diff ** 2)
#         sum = theano.printing.Print('sum')(T.sum(sqr, axis=1))
#         mean = theano.printing.Print('mean')(T.mean(sum))
#         return mean

        return T.mean(
                T.sum(
                (self.correct_output - self.predicted_output) ** 2, axis=1))

    def errors(self):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if self.correct_output.ndim != self.predicted_output.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                    ('correct_output', self.correct_output.type,
                    'predicted_output', self.predicted_output.type))
        # check if y is of the correct datatype
        if self.correct_output.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.predicted_output, self.correct_output))
        elif self.correct_output.dtype.startswith('float'):
            # First compare the equality of the data the same way numpy.allclose
            # does, then counts the failures.
            return T.sum(T.neq(0, T.sum(
                    T.le(self.predicted_output - self.correct_output,
                    1e-05 + 1e-08 * T.abs_(self.correct_output)), axis=1)))
        else:
            raise NotImplementedError()


class LblSoftmax(LossLayer):
    """
    Calculate the cross entropy of the softmax given as Equation 2 in
    [MniHin08].
    """

    def __init__(self, name='LblSoftmax'):
        super(LblSoftmax, self).__init__(name=name)

    def link(self, inputs):
        """

        Parameters
        ----------
        inputs
            0) predicted network output of all examples in the batch (matrix
                batch size x embeddings size)
            1) indices of the correct embeddings of all examples in the batch
                (vector: batch size)
            2) all embeddings (matrix vocabulary size x embeddings size)

        Returns
        -------
        []
            0) see cost()
            1) see errors()
        """
#         self.inputs = inputs
#         self.dot_product = theano.printing.Print('dot_product')(self.inputs[0])
#         self.target_indices = theano.printing.Print('target_indices')(self.inputs[1])
#         self.outputs = list(self.cost())
#         self.prediction = list(self.predict())

        self.inputs = inputs
        self.dot_product = self.inputs[0]
        self.target_indices = self.inputs[1]
        self.outputs = list(self.cost())
        self.prediction = list(self.predict())
        return self.outputs

    def cost(self):
        """Calculate the average cost per example."""
#         diff = theano.printing.Print('difference')(self.correct_output - self.predicted_output)
#         sqr = theano.printing.Print('square')(diff ** 2)
#         sum = theano.printing.Print('sum')(T.sum(sqr, axis=1))
#         mean = theano.printing.Print('mean')(T.mean(sum))
#         return mean

#         nll, softmax, argmax = T.nnet.crossentropy_softmax_1hot(
#                 self.dot_product, self.target_indices)
        nll, softmax = T.nnet.crossentropy_softmax_1hot(
                self.dot_product, self.target_indices)

#         nll = theano.printing.Print('nll')(nll)
#         softmax = theano.printing.Print('softmax')(softmax)

        return T.mean(nll), nll, softmax

    def predict(self):
        sm = T.nnet.softmax(self.dot_product)
        return sm, T.argmax(sm, axis=1)

class HammingLoss(SquaredErrorLossLayer):
    """Computes the distance between two vectors in terms of Hamming distance.
    """

    def __init__(self, name='HammingLossLayer'):
        super(HammingLoss, self).__init__(name=name)

    def cost(self):
        """Calculate the average cost per example."""
#         neq = theano.printing.Print('not equal')(T.neq(self.correct_output, self.predicted_output))
#         sum = theano.printing.Print('sum')(T.sum(neq, axis=1))
#         mean = theano.printing.Print('mean')(T.mean(sum))
#         return mean

        return T.mean(
                T.sum(
                T.neq(self.correct_output, self.predicted_output), axis=1))

#         return binary_crossentropy(self.predicted_output, self.correct_output).mean()


# class ElementWiseCrossEntropyLoss(SquaredErrorLossLayer):
#     """Compute the element wise cross-entropy error."""
#
#     def __init__(self, name='CrossEntropyLossLayer'):
#         super(ElementWiseCrossEntropyLoss, self).__init__(name=name)
#
#     def cost(self):
#         pred = theano.printing.Print('pred_output')(self.predicted_output)
#         corr = theano.printing.Print('corr_output')(self.correct_output)
#         cross_entr = theano.printing.Print('cross_entropy')(
#                 T.nnet.binary_crossentropy(pred,
#                 corr))
#         return cross_entr
#
# #         return T.nnet.binary_crossentropy(self.predicted_output,
# #                 self.correct_output)


class IndependendAttributesLoss(SquaredErrorLossLayer):
    """Computes the errors for multiple independent attributes.

    This is Equation 6.145 in Bis95 (Neural Networks for Pattern Recognition).

    $E=-\sum_n \sum_{k=1}^{c} \left\{t_k^n \ln y_k^n + (1-t_k^n) \ln(1-y_k^n)
    \right\} $
    , where t is the target, y the net's output, n the number of examples, c
    the number of individual outputs
    """

    def __init__(self, name='CrossEntropyLossLayer'):
        super(IndependendAttributesLoss, self).__init__(name=name)

    def cost(self):
#         pred = theano.printing.Print('pred_output')(self.predicted_output)
#         corr = theano.printing.Print('corr_output')(self.correct_output)
#         first = theano.printing.Print('first')(corr * T.log(pred))
#         second = theano.printing.Print('second')((1 - corr) * T.log(1 - pred))
#         err = theano.printing.Print('error')(-T.sum(first + second))
#         return err

        # Note: binary_crossentropy already gives the negative value, so we
        # don't have to take the negative sum here.
        return T.sum(binary_crossentropy(self.predicted_output, self.correct_output))

#         return -T.sum(self.correct_output * T.log(self.predicted_output) +
#                 (1. - self.correct_output) * T.log(1. - self.predicted_output))


class HingeLayer(LossLayer):
    """ Computes hinge loss of the correct and corrupted samples."""

    def __init__(self, name='HingeLayer'):
        super(HingeLayer, self).__init__(name=name)

    def link(self, inputs):
        """

        Parameters
        ----------
        inputs
            0) scores of the positive examples
            1) scores of the negative examples
        """
        self.inputs = inputs
        self.positive_score = self.inputs[0]
        self.negative_score = self.inputs[1]

        # Hinge loss
        self.scores = (T.ones_like(self.positive_score) -
                                     self.positive_score + self.negative_score)
        error = T.mean(self.scores * (self.scores > 0))

        self.cost = error
        self.outputs = [self.cost, error]
        return self.outputs

    def build(self):
        self.forward_pass = theano.function(inputs=self.inputs, outputs=self.outputs)

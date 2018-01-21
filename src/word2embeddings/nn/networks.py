#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Defines basic network structure that composed by several layers."""

from itertools import izip
from logging import getLogger

from theano import config, sparse, tensor as T
import theano

from cis.deep.utils import logger_config
from cis.deep.utils.theano import debug_print
import numpy as np
from word2embeddings.nn.layers import BiasedHiddenLayer, SerializationLayer, \
    IndependendAttributesLoss, SquaredErrorLossLayer
from word2embeddings.nn.util import zero_value, random_value_normal, \
    random_value_GloBen10
from word2embeddings.tools.theano_extensions import MRG_RandomStreams2


floatX = config.floatX

log = getLogger(__name__)
logger_config(log)

np.set_printoptions(edgeitems=100)


def get_activation_func(activation_func, x):
    """Returns a Theano activation function based on the given parameter.

    Parameters
    ----------
    activation_func : string
        activation function of neurons, choices: sigmoid, tanh, rect, softsign
    x : Theano variable
        data

    Returns
    -------
    Theano function
        activation function
    """

    if activation_func == 'tanh':
        return T.tanh(x)
    if activation_func == 'sigmoid':
        return T.nnet.sigmoid(x)
    if activation_func == 'rect':
        return T.maximum(0, x)
    if activation_func == 'softsign':
        from theano.sandbox.softsign import softsign
        return softsign(x)

    raise ValueError('unknown activation function: %s' % activation_func)


class Network(SerializationLayer):
    """ General full connected neural network."""
    def __init__(self, name='Network', inputs=None):
        self.name = name
        self.layers = []
        self.inputs = inputs
        self.exceptions = {}
        self._outputs = None

    def set_learning_rate(self, lr, method='global', lr_adaptation_method=None):
        """Set network global lr and update the local lr of the layers"""
        for layer in self.layers:
            layer.set_learning_rate(lr, method, lr_adaptation_method)

    def update_learning_rate(self, remaining):
        for layer in self.layers:
            layer.update_learning_rate(remaining)

    def size(self):
        return T.sum([layer.size() for layer in self.layers])

    def append(self, layer):
        """ Link a new layer to the stack of the hidden layers."""
        self.layers.append(layer)

        if len(self.layers) > 1:
            last = self.layers[-1]
            previous = self.layers[-2]
            last.link(previous.outputs)
        else:
            self.layers[0].link(self.inputs)

    def set_layers(self, layers):
        self.layers = layers

    def link(self, inputs):
        for layer in self.layers:
            layer.link(inputs)
            inputs = layer.outputs

    @property
    def cost(self):
        return self.layers[-1].outputs[0]

    @property
    def outputs(self):
        """ The output of the last layer in the network."""
        if not self._outputs:
            return self.layers[-1].outputs
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value

    def params(self):
        """ Iterates over all the parameters of the network."""
        for layer in self.layers:
            for param in layer.params():
                yield param

    def build(self):
        """ Build theano functions for training, validation and testing."""
#         self.forward_pass = theano.function(inputs=self.inputs,
#                 outputs=self.outputs)
        log.debug('## ' + str(self.cost))
        log.debug('## ' + str(self.inputs))

#         inputs = []
#         inputs.append(theano.printing.Print('labels', self.inputs[0]))
#         inputs.append(theano.printing.Print('input_data', self.inputs[1]))
#         cost = theano.printing.Print('cost', self.cost)

#         self.trainer = theano.function(inputs=inputs, outputs=cost,
#                 updates=self.updates(cost))
        self.trainer = theano.function(inputs=self.inputs, outputs=self.cost,
                updates=self.updates(self.cost))

        self.validator = theano.function(inputs=self.inputs, outputs=self.cost)

    @property
    def L1(self):
        """ Calculates the sum of all the parameters weights of the network."""
        return T.sum([layer.L1 for layer in self.layers])

    @property
    def L2(self):
        """ Calculates the sum of the squared parameters weights of the
        network.
        """
        return T.sum([layer.L2 for layer in self.layers])

    def updates(self, cost):
        """ Defines the list of functions that update the network parameters."""
        updates = []
        for layer in filter(lambda x: x not in self.exceptions, self.layers):
            for update in layer.updates(cost):
                param, _ = update
                if param not in self.exceptions:
                    updates.append(update)
        return updates

    def set_clipping(self, threshold):
        for layer in self.layers:
            layer.clipping = True
            layer.threshold = threshold

    def status(self):
        for layer in self.layers:
            for param, status in layer.status():
                yield param, status

    def info(self):
        for layer in self.layers:
            for param, status in layer.status():
                log.debug(param.name + u'\t\t' + status)


class StackedBiasedHidden(Network):
    """ Sequence of fully connected layers."""

    def __init__(self, name='Biased', layers=None, **kwargs):
        super(StackedBiasedHidden, self).__init__()
        self.name = name
        self.stack_layers(layers, **kwargs)

    def stack_layers(self, layers, **kwargs):
        """ Create a list of fully connected hidden layers according to the
        sizes.
        """
        shapes = izip(layers[:-1], layers[1:])
        for i, shape in enumerate(shapes):
            name = '%s_layers_stack_%i_(%ix%i)' % (self.name, i, shape[0], shape[1])
            layer = BiasedHiddenLayer(name=name, shape=shape, **kwargs)
            self.layers.append(layer)

            log.debug('---------- ' + str(type(layer)) + str(layer.__dict__))

    def link(self, inputs):
        """

        Parameters
        ----------
        inputs
            0) input for the first hidden layer
        """
        self.inputs = inputs

        # Set the input of the current hidden layer to the output of the
        # previous hidden layer.
        for layer in self.layers:
            inputs = layer.link([inputs[0]])

        return self.outputs

    def build(self):
#         self.forward_pass = theano.function(inputs=self.inputs,
#                 outputs=self.outputs)
        pass


class Model(Network):
    total_examples = 0
    total_epochs = 0
    total_costs = 0.0


class MultiLayerPerceptron(Model):
    """Standard fully connected feed forward net.

    Parameters
    ----------
    input_size : (int, int)
        number of input neurons
    output_size : int
        number of output neurons
    hidden_layers : list(int)
        number of neurons for each hidden layer
    error_func : str
        name of the error function to be used; either cross_entropy or
        least_squares
    activation_func : function
        activation function of neurons
    """

    def __init__(self, name='MultiLayerPerceptron', input_size=None,
            output_size=None, hidden_layers=[1], error_func='least_squares',
            activation_func=T.nnet.sigmoid):
        super(MultiLayerPerceptron, self).__init__(name=name)
        layers = [input_size] + hidden_layers + [output_size]
        self.hidden_stack = StackedBiasedHidden(name='w_stack', layers=layers,
                activation=activation_func)

        if error_func == 'cross_entropy':
            self.loss = IndependendAttributesLoss(name='loss')
            log.info('error function is cross-entropy')
        else:
            self.loss = SquaredErrorLossLayer(name='loss')
            log.info('error function is least squares')

        self.layers = [self.hidden_stack, self.loss]

    def build(self):
        super(MultiLayerPerceptron, self).build()

#         self.predictor = theano.function(inputs=[self.inputs[1]],
#                  outputs=theano.printing.Print('net_output')(self.hidden_stack.outputs[0]))

        self.predictor = theano.function(inputs=[self.inputs[1]],
                 outputs=self.hidden_stack.outputs[0])

    def link(self, inputs):
        """

        Parameters
        ----------
        inputs
            0) correct labels of the input data
            1) input data

        Returns
        -------
        []
            = LossLayer.link()
        """
        self.inputs = inputs
        labels = inputs[0]
        data = inputs[1]
        hidden_output = self.hidden_stack.link([data])[0]
        self.outputs = self.loss.link([hidden_output, labels])
        return self.outputs


class SimpleVLblNce(SerializationLayer):
    """This is the position independent vLBL model from [MniKav13].

    We train the model using noise-contrastive estimation.
    """
    total_examples = 0
    total_epochs = 0
    total_costs = 0.0

    keep_as_ndarray = ['unigram']

    # Put all parameters in here that needs to be updated during training. The
    # updates are done automatically.
    updatable_parameters = ['R', 'Q', 'bias']

    # List of Theano variable names that will be considered in regularization
    # computation.
    regularize = ['R', 'Q']

    def __init__(self, batch_size, vocab_size, left_context, right_context,
            emb_size, k, unigram, l1_weight=0, l2_weight=0, nce_seed=2345):
        self.name = 'vLBL'
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.left_context = left_context
        self.right_context = right_context
        self.context_size = self.left_context + self.right_context
        self.emb_size = emb_size
        self.k = k
        self.unigram = unigram
        self.p_n = debug_print(theano.shared(value=unigram, name='noise_probab'),
                'noise')

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.nce_seed = nce_seed

        # Create context and target embeddings
        rand_values = random_value_normal((self.vocab_size, self.emb_size),
                floatX, np.random.RandomState(1234))
        self.R = theano.shared(value=rand_values, name='R')
        rand_values = random_value_normal((self.vocab_size, self.emb_size),
                floatX, np.random.RandomState(4321))
        self.Q = theano.shared(value=rand_values, name='Q')
        b_values = zero_value((self.vocab_size,), dtype=floatX)
        self.bias = theano.shared(value=b_values, name='bias')

        # The learning rates are created the first time set_learning_rate is
        # called.
        self.lr = None

    def build(self):
        pass

    def _calc_q_hat(self, r_h):
        """Calculates q_hat according to the model.

        Parameters
        ----------
        r_h : ndarray
            embedded context words
        c : ndarray
            context dependent weight vectors; not used by this model

        Returns
        -------
        ndarray (batch_size x emb_size)
            q_hat as calculated in Eq. 2 in [MniKav13] without C
        """
        return T.mean(r_h, axis=1)

    def _calc_r_h(self, h_indices):
        """Calculates r_h according to the model.

        Parameters
        ----------
        h_indices : ndarray
            array with indices of context words
            [batch_size] * [context_size]

        Returns
        -------
        ndarray (batch_size x context_size x emb_size)
            r_h used as an input in Eq. 2 in [MniKav13]
        """
        return self._embed_context(h_indices)

    def _calc_regularization_cost(self):
        """Calculate the regularization cost given the weight decay parameters.

        Only the parameters will be considered that are stored in the set
        self.regularize.

        Returns
        -------
        theano variable
            regularization cost depending on the parameters to be regularized
            and the weight decay parameters for L1 and L2 regularization.
        """
        cost = theano.shared(value=np.cast[floatX](.0))
        l1_cost = 0
        l2_cost = 0

        for p in self.regularize:
            l1_cost += T.sum(T.abs_(self.__dict__[p]))
            l2_cost += T.sum(T.sqr(self.__dict__[p]))

        l1_cost = debug_print(l1_cost, 'l1_cost')
        l2_cost = debug_print(l2_cost, 'l2_cost')

        if self.l1_weight != 0:
            cost += self.l1_weight * l1_cost

        if self.l2_weight != 0:
            cost += self.l2_weight * l2_cost

        return cost

    def _create_adagrad_matrices(self):
        """Create AdaGrad gradient matrices for all updatable variables.

        New matrices are only created if they don't exist yet, i.e., if a model
        was loaded, the already existing matrices are not overwritten.
        the matrices contain the sum of squared gradients for all parameters in
        the model.
        Caution: Do not put the matrices into a dict. In that case pickling does
        not work.
        """

        for p in self.updatable_parameters:
            name = 'adagrad_matrix_' + p

            # Have we already created the gradient matrices? This might happen after
            # we loaded a model.
            if name in self.__dict__:
                continue

            self.__dict__[name] = \
                    theano.shared(np.zeros(self.__dict__[p].shape.eval(),
                    dtype=floatX), name='adagrad_matrix_' + p)

    def _embed_context(self, indices):
        """Embed the context indices.

        Parameters
        ----------
        indices : ndarray(batch_size x context_size)
            indices to be embedded

        Returns
        -------
        ndarray(batch_size x context_size x emb_size)
            embedded context
        """
        embedded = self._embed_word_indices(indices, self.R)
        concatenated_rows = embedded.flatten()
        example_count = indices.shape[0]
        return concatenated_rows.reshape((example_count, self.context_size,
                self.emb_size))

    def _embed_noise(self, indices):
        """Embed the noise indices.

        Parameters
        ----------
        indices : ndarray(batch_size x noise_samples)
            indices to be embedded

        Returns
        -------
        ndarray(batch_size x noise_samples x emb_size)
            embedded context
        """
        embedded = self._embed_word_indices(indices, self.Q)
        concatenated_rows = embedded.flatten()
        example_count = indices.shape[0]
        return concatenated_rows.reshape((example_count, self.k,
                self.emb_size))

    def _embed_target(self, indices):
        """Embed the target indices.

        Parameters
        ----------
        indices : ndarray(batch_size x 1)
            indices to be embedded

        Returns
        -------
        ndarray(batch_size x emb_size)
            embedded context
        """
        embedded = self._embed_word_indices(indices, self.Q)
        num_examples = indices.shape[0]
        emb_size = embedded.size // num_examples
        return embedded.reshape((num_examples, emb_size))

    def _embed_word_indices(self, indices, embeddings):
        """Embed all indexes using the given embeddings.

        Parameters
        ----------
        indices : ndarray
            indices of the items to embed using the given embeddings matrix
            Note: indices are flattened
        embeddings : ndarray (vocab_size x emb_size)
            embeddings matrix

        Returns
        -------
        ndarray (len(indices) x emb_size)
            embedded indices
        """
        concatenated_input = indices.flatten()

        # Rami's fix
        if config.device == 'gpu':
            embedded = embeddings[concatenated_input]
        else:
            embedded = theano.sparse_grad(embeddings[concatenated_input])

        return embedded

    def _get_gradients(self, J):
        """Get all gradients needed by the model.

        Parameters
        ----------
        J : theano function
            cost function

        Returns
        -------
        (float, ...)
            all gradients of the model
        """
        return T.grad(J, [self.__dict__[p] for p in self.updatable_parameters])

    def _get_gradients_adagrad(self, J):
        """Get the AdaGrad gradients and squared gradients updates.

        The returned gradients still need to be multiplied with the general
        learning rate.

        Parameters
        ----------
        J : theano variable
            cost

        Returns
        -------
        theano variable
            gradients that are adapted by the AdaGrad algorithm
        theano variable
            updated sum of squares for all previous steps
        """
        grads = T.grad(J, [self.__dict__[self.updatable_parameters[i]]
                for i in xrange(len(self.updatable_parameters))])

        for i, _ in enumerate(grads):
            grads[i] = debug_print(grads[i], 'grads_' + self.updatable_parameters[i])

        updated_squares = dict()

        # Add squared gradient to the squared gradient matrix for AdaGrad and
        # recalculate the gradient.
        for i, p in enumerate(self.updatable_parameters):

            # We need to handle sparse gradient variables differently
            if isinstance(grads[i], sparse.SparseVariable):
                # Add the sqares to the matrix
                power = debug_print(sparse.structured_pow(grads[i], 2.), 'pow_' + p)
                # Remove zeros (might happen when squaring near zero values)
                power = sparse.remove0(power)
                updated_squares[p] = self.__dict__['adagrad_matrix_' + p] + power

                # Get only those squares that will be altered, for all others we
                # don't have gradients, i.e., we don't need to consider them at
                # all.
                sqrt_matrix = sparse.sp_ones_like(power)
                sqrt_matrix = debug_print(updated_squares[p] * sqrt_matrix, 'adagrad_squares_subset_' + p)

                # Take the square root of the matrix subset.
                sqrt_matrix = debug_print(sparse.sqrt(sqrt_matrix), 'adagrad_sqrt_' + p)
                # Calc 1. / the square root.
                sqrt_matrix = debug_print(sparse.structured_pow(sqrt_matrix, -1.), 'adagrad_pow-1_' + p)
                grads[i] = sparse.mul(grads[i], sqrt_matrix)
            else:
                power = debug_print(T.pow(grads[i], 2.), 'pow_' + p)
                updated_squares[p] = self.__dict__['adagrad_matrix_' + p] + power

                # Call sqrt only for those items that are non-zero.
                denominator = T.switch(T.neq(updated_squares[p], 0.0),
                        T.sqrt(updated_squares[p]),
                        T.ones_like(updated_squares[p], dtype=floatX))
                grads[i] = T.mul(grads[i], 1. / denominator)

            updated_squares[p] = debug_print(updated_squares[p], 'upd_squares_' + p)

        for i, _ in enumerate(grads):
            grads[i] = debug_print(grads[i], 'grads_updated_' + self.updatable_parameters[i])

        return grads, updated_squares

    def get_learning_rate(self):
        """Return the learning rate(s).

        Returns
        -------
        dict
            returns dict, containing the different learning rates for different
            parameters, e.g., embeddings.
        """
        return {k: self.lr[k].get_value() for k in self.lr}


    def _get_noise(self, batch_size):
        # Create unigram noise distribution.
        srng = MRG_RandomStreams2(seed=self.nce_seed)

        # Get the indices of the noise samples.
        random_noise = debug_print(srng.multinomial(
                size=(batch_size, self.k), pvals=self.unigram), 'random_noise')

        noise_indices_flat = debug_print(random_noise.reshape(
                (batch_size * self.k,)), 'noise_indices_flat')
        p_n_noise = debug_print(self.p_n[noise_indices_flat].reshape(
                (batch_size, self.k)), 'p_n_noise')
        return random_noise, p_n_noise

    def link(self, inputs):
        self.inputs = inputs
        h_indices = inputs[0]
        w_indices = inputs[1]

        if self.lr_adaptation_method == 'adagrad':
            self._create_adagrad_matrices()

        # Embed context and target words
        r_h = debug_print(self._calc_r_h(h_indices),
                'embed_context')
        q_w = debug_print(self._embed_target(w_indices), 'embed_target')

        # This is the actual, not the expected batch size. Since there might be
        # batches that are not complete, e.g., at the end of a dataset, this can
        # differ.
        batch_size = q_w.shape[0]

        # Calculate predicted target word embedding
        q_hat = debug_print(self._calc_q_hat(r_h), 'q_hat')

        noise_indices, p_n_noise = self._get_noise(batch_size)

        # Get the data part of Eq. 8.
        s_theta_data = debug_print(
                T.sum(q_hat * q_w, axis=1) + self.bias[w_indices],
                's_theta_data')
        p_n_data = debug_print(self.p_n[w_indices], 'p_n_data')
        delta_s_theta_data = debug_print(
                s_theta_data - T.log(self.k * p_n_data),
                'delta_s_theta_data')

        log_sigm_data = debug_print(T.log(T.nnet.sigmoid(delta_s_theta_data)),
                'log_sigm_data')

        # Get the data noise of Eq. 8.
        q_noise = debug_print(self._embed_noise(noise_indices), 'q_noise')
        q_hat_res = q_hat.reshape((batch_size, 1, self.emb_size))
        s_theta_noise = debug_print(
                T.sum(q_hat_res * q_noise, axis=2) + self.bias[noise_indices],
                's_theta_noise')
        delta_s_theta_noise = debug_print(s_theta_noise - T.log(self.k * p_n_noise),
                'delta_s_theta_noise')

        log_sigm_noise = debug_print(T.log(1 - T.nnet.sigmoid(delta_s_theta_noise)),
                'log_sigm_noise')
        sum_noise_per_example = debug_print(T.sum(log_sigm_noise, axis=1),
                'sum_noise_per_example')

        # Calc objective function (cf. Eq. 8 in [MniKav13])
        # [MniKav13] maximize, therefore we need to switch signs in order to
        # minimize
        J = debug_print(-T.mean(log_sigm_data) - T.mean(sum_noise_per_example),
                'J')
        regularization_cost = debug_print(self._calc_regularization_cost(),
                'regularization_cost')
        self.cost = debug_print(J + regularization_cost, 'overall_cost')

        self.outputs = [h_indices, w_indices, q_hat, self.cost]
        self.outputs.extend(self.updatable_parameters)

        if self.lr_adaptation_method == 'adagrad':
            grads, updated_squares = self._get_gradients_adagrad(self.cost)

            # Update the running squared gradient for AdaGrad
            additional_updates = [(self.__dict__['adagrad_matrix_' + p],
                    updated_squares[p])
                    for p in self.updatable_parameters]
        else:
            grads = self._get_gradients(self.cost)
            additional_updates = []

        updates = []

        try:
            for i, p in enumerate(self.updatable_parameters):
                updates.append((self.__dict__[p],
                        self.__dict__[p] -
                        self.lr.get(p, self.lr['default']) * grads[i]))
        except KeyError:
            raise ValueError('Not all parameters or "default" specified in ' +
                    'the learning-rate parameter.')

        updates.extend(additional_updates)

        self.trainer = theano.function([h_indices, w_indices],
                [self.cost, h_indices, w_indices], updates=updates)

        # In prediction we have to normalize explicitly in order to receive a
        # real probability distribution.
        s_theta_all_w = T.dot(q_hat, self.Q.T) + self.bias

        self.predictor = theano.function([h_indices],
                [q_hat, s_theta_all_w])
        self.validator = theano.function([h_indices, w_indices],
                [self.cost, s_theta_all_w])
        return self.cost

    def update_learning_rate(self, remaining):
        """Update the learning rate depending on a given method."""
        new_value = self.global_lr

        if self.lr_adaptation_method == 'linear':
            new_value = {k: v * remaining for k, v in new_value.iteritems()}

        # Don't change anything if method is 'adagrad', because the general
        # learning rate doesn't change.
        # Don't change anything if the method is 'constant'

        # Here we need to handle cases depending on the number of learning
        # rates.

        for k, v in new_value.iteritems():
            self.lr[k].set_value(v)

        log.debug('Param %s\'s learning rate is %s' %
                (self.name, str(new_value)))

    def set_learning_rate(self, lr, method='global',
            lr_adaptation_method='constant'):
        """Sets the learning rate and the method to the given values.

        Currently, only 'global' is valid as method and None or 'linear' are
        valid as lr_adaptation_method.
        """

        if method not in ('global'):
            raise ValueError(method + ' is not accepted as learning method. ' +
                    'Currently, only "global" is valid as method.')

        # The learning rate variables cannot be created in the __init__ method,
        # because there may be different learning rates for different parameters
        # which are not given to the constructor. Putting it there would create
        # duplicate code with this method.
        if not self.lr:
            self.lr = {k:
                    theano.shared(value=np.cast[floatX](1.0), name='lr_' + k)
                    for k in lr}

        self.global_lr = lr
        self.lr_adaptation_method = lr_adaptation_method
        self.update_learning_rate(1.0)


class VLblNce(SimpleVLblNce):
    updatable_parameters = ['R', 'Q', 'bias', 'C']

    def __init__(self, *args, **kwargs):
        super(VLblNce, self).__init__(*args, **kwargs)

        self.regularize.append('C')

        rand_values = random_value_GloBen10((self.context_size, self.emb_size),
                floatX, np.random.RandomState(2341),
                (self.context_size * self.emb_size, self.emb_size))
        self.C = theano.shared(value=rand_values, name='C')

    def _calc_q_hat(self, r_h):
        """Calculates q_hat according to the model.

        Parameters
        ----------
        r_h : ndarray
            embedded context words

        Returns
        -------
        ndarray
            q_hat as calculated in Eq. 2 in [MniKav13]
        """
        inner = debug_print(r_h * self.C, 'inner-q_hat')
        return T.sum(inner, axis=1)


class NvLblNce(VLblNce):
    """Implementation of a non-linear log-bilinear language model."""

    def __init__(self, activation_func='tanh', *args, **kwargs):
        """
        Parameters
        ----------
        activation_func : string
            activation function of neurons, choices: sigmoid, tanh, rect, softsign
        """
        super(NvLblNce, self).__init__(*args, **kwargs)
        self.name = 'nvlbl'
        self.activation_func = activation_func

    def _calc_q_hat(self, r_h):
        """Calculates q_hat according to the model vLBL model but adds a non-
        linearity.

        Parameters
        ----------
        r_h : ndarray
            embedded context words

        Returns
        -------
        ndarray
            q_hat as calculated in Eq. 2 in [MniKav13] with a non-linearity
            around it.
        """
        return get_activation_func(self.activation_func,
                super(NvLblNce, self)._calc_q_hat(r_h))


class LblNce(SimpleVLblNce):
    """This is an implementation of the original LBL matrix model using NCE."""

    def __init__(self, *args, **kwargs):
        super(LblNce, self).__init__(*args, **kwargs)
        self.name = 'lbl'
        self.updatable_parameters.append('W')
        self.regularize.append('W')

        rand_values = random_value_GloBen10(
                (self.context_size * self.emb_size, self.emb_size),
                floatX, np.random.RandomState(7816))
        self.W = theano.shared(value=rand_values, name='W')

    def _calc_q_hat(self, r_h):
        """Calculates q_hat according to the model.

        Parameters
        ----------
        r_h : ndarray
            embedded context words

        Returns
        -------
        ndarray
            q_hat as calculated in Eq. 2 in [MniTeh12]
        """
        return T.dot(r_h, self.W)

    def _embed_context(self, indices):
        """Embed the context indices.

        Parameters
        ----------
        indices : ndarray(batch_size x context_size)
            indices to be embedded

        Returns
        -------
        ndarray(batch_size x (context_size * emb_size))
            embedded context
        """
        embedded = self._embed_word_indices(indices, self.R)
        concatenated_rows = embedded.flatten()
        example_count = indices.shape[0]
        return concatenated_rows.reshape((example_count, self.context_size *
                self.emb_size))


class NlblNce(LblNce):
    """This is an implementation of a non-linear version of the LBL matrix model
    using NCE.
    """

    def __init__(self, activation_func='tanh', *args, **kwargs):
        super(NlblNce, self).__init__(*args, **kwargs)
        self.name = 'nlbl'
        self.activation_func = activation_func

    def _calc_q_hat(self, r_h):
        """Calculates q_hat according to the model.

        Parameters
        ----------
        r_h : ndarray
            embedded context words

        Returns
        -------
        ndarray
            q_hat as calculated in Eq. 2 in [MniTeh12] and surrounded by a non-
            linearity
        """
        lbl_q_hat = super(NlblNce, self)._calc_q_hat(r_h)
        return get_activation_func(self.activation_func, lbl_q_hat)


class SLmNce(SimpleVLblNce):
    """This is an implementation of a shallow neural network language model
    (SLM) using NCE.
    """

    def __init__(self, hidden_neurons=100, activation_func='tanh', *args,
            **kwargs):
        super(SLmNce, self).__init__(*args, **kwargs)
        self.name = 'slm'
        self.activation_func = activation_func
        self.hidden_neurons = hidden_neurons
        self.updatable_parameters.extend(['W1', 'W2'])

        rand_values = random_value_GloBen10(
                (self.context_size * self.emb_size + 1, self.hidden_neurons),
                floatX, np.random.RandomState(7816))
        self.W1 = theano.shared(value=rand_values, name='W')
        rand_values = random_value_GloBen10(
                (self.hidden_neurons + 1, self.emb_size),
                floatX, np.random.RandomState(7817))
        self.W2 = theano.shared(value=rand_values, name='W')

    def _calc_q_hat(self, r_h):
        """Calculates q_hat according to the model.

        Parameters
        ----------
        r_h : ndarray
            embedded context words

        Returns
        -------
        ndarray
            q_hat, computed by a single hidden layer
        """
        bias = debug_print(T.ones((r_h.shape[0], 1), dtype=floatX), 'bias1')
        r_h = debug_print(T.concatenate([r_h, bias], axis=1), 'r_h_concatenate')
        hidden_output = debug_print(get_activation_func(self.activation_func,
                T.dot(r_h, self.W1)), 'hidden_output')
        bias = debug_print(T.ones((hidden_output.shape[0], 1), dtype=floatX), 'bias2')
        hidden_output = debug_print(T.concatenate([hidden_output, bias], axis=1), 'hidden_output_concatenate')
        return get_activation_func(self.activation_func,
                T.dot(hidden_output, self.W2))

    def _calc_regularization_cost(self):
        """Calculate the regularization cost given the weight decay parameters.

        Only the parameters will be considered that are stored in the set
        self.regularize. We need to handle it manually in this class, because
        the weight matrices contain bias columns, which should not be considered
        in regularization computation. Therefore, do not!!! add W1 and W2 to
        self.regularize

        Returns
        -------
        theano variable
            regularization cost depending on the parameters to be regularized
            and the weight decay parameters for L1 and L2 regularization.
        """
        cost = super(SLmNce, self)._calc_regularization_cost()
        l1_cost = T.sum(T.abs_(self.W1[:, :-1]))
        l1_cost += T.sum(T.abs_(self.W2[:, :-1]))
        l2_cost = T.sum(T.sqr(self.W1[:, :-1]))
        l2_cost += T.sum(T.sqr(self.W2[:, :-1]))

        if self.l1_weight != 0:
            cost += self.l1_weight * l1_cost

        if self.l2_weight != 0:
            cost += self.l2_weight * l2_cost

        return cost

    def _embed_context(self, indices):
        """Embed the context indices.

        Taken from NlblNce.

        Parameters
        ----------
        indices : ndarray(batch_size x context_size)
            indices to be embedded

        Returns
        -------
        ndarray(batch_size x (context_size * emb_size))
            embedded context
        """
        embedded = self._embed_word_indices(indices, self.R)
        concatenated_rows = embedded.flatten()
        example_count = indices.shape[0]
        return concatenated_rows.reshape((example_count, self.context_size *
                self.emb_size))


class VLblNceDistributional(VLblNce):

    def __init__(self, *args, **kwargs):
        super(VLblNceDistributional, self).__init__(*args, **kwargs)

    def load_params(self, base_filename, params_str):
        super(VLblNceDistributional, self).load_params(base_filename, params_str)

        #if 'D' in self.__dict__:
        #    self.D2 = theano.sparse.basic.dense_from_sparse(self.D)

        # do not re-initialize R if it is loaded - so here we check whether
        #dimensions of R are right
        if 'R' not in self.__dict__ \
            or self.R.shape.eval()[0] != self.D.shape.eval()[1] :

            rand_values = random_value_normal((self.D.shape.eval()[1], \
                self.emb_size), floatX, np.random.RandomState(999))
            self.R = theano.shared(value=rand_values, name='R')

    def _embed_distrib_context(self, indices):
        """Embed the context indices.

        Parameters
        ----------
        indices : ndarray(batch_size x context_size)
            indices to be embedded

        Returns
        -------
        ndarray(batch_size x context_size x emb_size)
            embedded context
        """
        embedded = self._embed_word_indices(indices, self.D)
        try:
            concatenated_rows = embedded.flatten()
        except AttributeError:
            concatenated_rows = theano.sparse.basic.dense_from_sparse(embedded).flatten()

        example_count = indices.shape[0]

        return concatenated_rows.reshape((example_count, self.context_size,
                self.D.shape.eval()[1]))

    def _calc_r_h(self, h_indices):
        """Calculates r_h according to the model.

        Parameters
        ----------
        h_indices : ndarray
            array with indices of context words
            [batch_size] * [context_size]

        Returns
        -------
        ndarray
            r_h used as an input in Eq. 2 in [MniKav13]
        """
        if 'D' not in self.__dict__:
            raise ValueError('Parameter D is not specified for a model. ' +
                    'Please, check your training script.')
        emb_d = self._embed_distrib_context(h_indices)

        return T.dot(emb_d, self.R)

    def _embed_word_indices(self, indices, embeddings):
        """Embed all indexes using the given embeddings.

        Parameters
        ----------
        indices : ndarray
            indices of the items to embed using the given embeddings matrix
            Note: indices are flattened
        embeddings : ndarray (vocab_size x emb_size)
            embeddings matrix

        Returns
        -------
        ndarray (len(indices) x emb_size)
            embedded indices
        """
        try:
            embedded = super(VLblNceDistributional, self) \
                ._embed_word_indices(indices, embeddings)
        except ValueError:
            concatenated_input = debug_print(indices.flatten(), "concatenated_input")

            # Commented till new version of Theano is available
            #embedded = theano.sparse.basic.get_item_list(embeddings, concatenated_input)

            # Fix for current Theano's version: get rows of embeddings matrix
            #via multiplying it with other sparse matrix
            data = debug_print(T.ones_like(T.arange(concatenated_input.shape[0])), 'data')
            ind = debug_print(concatenated_input, 'ind')
            indptr = debug_print(T.arange(concatenated_input.shape[0] + 1), 'indptr')
            shape = debug_print(T.stack(concatenated_input.shape[0], embeddings.shape[0]), 'shape')
            mult1 = debug_print(theano.sparse.basic.CSR(data, ind, indptr, shape), 'mult1')

            embedded =  debug_print(theano.sparse.basic.structured_dot(mult1, embeddings), 'embedded')

        return embedded
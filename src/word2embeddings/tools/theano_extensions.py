# -*- coding: utf-8 -*-
import numpy
from theano import Op, Apply
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.basic import as_tensor_variable

import theano.tensor as T


class MRG_RandomStreams2(MRG_RandomStreams):
    """Module component with similar interface to numpy.random
    (numpy.random.RandomState)
    """

    def __init__(self, seed=12345, use_cuda=None):
        """
        :type seed: int or list of 6 int.

        :param seed: a default seed to initialize the random state.
            If a single int is given, it will be replicated 6 times.
            The first 3 values of the seed must all be less than M1 = 2147483647,
            and not all 0; and the last 3 values must all be less than
            M2 = 2147462579, and not all 0.

        """
        super(MRG_RandomStreams2, self).__init__(seed, use_cuda)

    def multinomial(self, size=None, n=1, pvals=None, ndim=None, dtype='int32',
                    nstreams=None):
        """
        Sample `n` (currently `n` needs to be 1) times from a multinomial
        distribution defined by probabilities pvals.

        Example : pvals = [[.98, .01, .01], [.01, .98, .01]] will
        probably result in [[1,0,0],[0,1,0]].

        .. note::
            -`size` and `ndim` are only there keep the same signature as other
            uniform, binomial, normal, etc.
            todo : adapt multinomial to take that into account

            -Does not do any value checking on pvals, i.e. there is no
             check that the elements are non-negative, less than 1, or
             sum to 1. passing pvals = [[-2., 2.]] will result in
             sampling [[0, 0]]
        """
        if pvals is None:
            raise TypeError('You have to specify pvals')
        pvals = as_tensor_variable(pvals)
        if size is not None:
            if any([isinstance(i, int) and i <= 0 for i in size]):
                raise ValueError(
                    'The specified size contains a dimension with value <= 0',
                    size)

        if n == 1 and pvals.ndim == 1:
            if ndim is not None:
                raise ValueError('Provided an ndim argument to ' +
                        'MRG_RandomStreams2.multinomial, which does not use ' +
                        'the ndim argument.')
            unis = self.uniform(size=size, ndim=2, nstreams=nstreams)
            op = MultinomialFromUniform2(dtype)
            return op(pvals, unis)
        else:
            raise NotImplementedError('MRG_RandomStreams2.multinomial only ' +
                ' implemented with n == 1 and pvals.ndim = 2')


class MultinomialFromUniform2(Op):
    '''Converts samples from a uniform into sample from a multinomial.

    This random number generator is faster than the standard one of Theano,
    because it stops earlier and doesn't return matrices of zeros and ones,
    indicating which index was drawn. Instead it returns the index of the drawn
    element.
    '''
    def __init__(self, odtype):
        self.odtype = odtype

    def __eq__(self, other):
        return type(self) == type(other) and self.odtype == other.odtype

    def __hash__(self):
        return hash((type(self), self.odtype))

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        try:
            self.odtype
        except AttributeError:
            self.odtype = 'auto'

    def make_node(self, pvals, unis):
        pvals = T.as_tensor_variable(pvals)
        unis = T.as_tensor_variable(unis)
        if pvals.ndim != 1:
            raise NotImplementedError('pvals ndim should be 1', pvals.ndim)
        if unis.ndim != 2:
            raise NotImplementedError('unis ndim should be 2', unis.ndim)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        out = T.tensor(dtype=odtype, broadcastable=unis.type.broadcastable)
        return Apply(self, [pvals, unis], [out])

    def grad(self, ins, outgrads):
        pvals, unis = ins
        (gz,) = outgrads
        return [T.zeros_like(x) for x in ins]

#     def c_code_cache_version(self):
#         return (5,)
 
    def c_code(self, node, name, ins, outs, sub):
        (pvals, unis) = ins
        (z,) = outs

        fail = sub['fail']
        return """
        if (PyArray_NDIM(%(pvals)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "pvals wrong rank");
            %(fail)s;
        }
        if (PyArray_NDIM(%(unis)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "unis wrong rank");
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || ((PyArray_DIMS(%(z)s))[0] != (PyArray_DIMS(%(unis)s))[0])
            || ((PyArray_DIMS(%(z)s))[1] != (PyArray_DIMS(%(unis)s))[1])
        )
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_ZEROS(2,
                PyArray_DIMS(%(unis)s),
                type_num_%(z)s,
                0);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }

        { // NESTED SCOPE

        const int nb_outcomes = PyArray_DIMS(%(pvals)s)[0];
        const int nb_rows = PyArray_DIMS(%(unis)s)[0];
        const int nb_cols = PyArray_DIMS(%(unis)s)[1];

        //
        // For each multinomial, loop over each possible outcome
        //
        for (int row = 0; row < nb_rows; ++row)
        {
            for (int col = 0; col < nb_cols; ++col) {
//                std::cout << row << 'x' << col << std::endl;

                dtype_%(pvals)s cummul = 0.;
                const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR2(%(unis)s, row, col);
                dtype_%(z)s* z_nm = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, row, col);
                *z_nm = -1;

//                std::cout << "unis " << (int)(*unis_n * 100) << std::endl;
//                std::cout << "z_nm " << (int)(*z_nm * 100) << std::endl;

                for (int m = 0; m < nb_outcomes; ++m)
                {
                    const dtype_%(pvals)s* pvals_m = (dtype_%(pvals)s*)PyArray_GETPTR1(%(pvals)s, m);
                    cummul += *pvals_m;
//                    std::cout << "cummul " << (int)(cummul * 100) << std::endl;

                    if (cummul > *unis_n)
                    {
                        *z_nm = m;
//                        *z_nm = 17;
                        break;
                    }

                }

                // If we reached the end, use the last value.
                // If we have a real distribution [0,1], than this should never
                // happen, right? I got a segmentation fault when removing it.
                // 2014-04-08
                // This might happen due to rounding errors. 2014-05-01
                if (*z_nm == -1) {
                    *z_nm = nb_outcomes - 1;
                }
            }
        }
        } // END NESTED SCOPE
        """ % locals()

    def perform(self, node, ins, outs):
        (pvals, unis) = ins
        (z,) = outs

        if z[0] is None or z[0].shape != numpy.sum(unis.shape):
            z[0] = numpy.zeros(unis.shape, dtype=node.outputs[0].dtype)

        z[0][:, :] = -1

        nb_outcomes = pvals.shape[0]

        for row in xrange(unis.shape[0]):
            for col in xrange(unis.shape[1]):
                cummul = 0
                unis_n = unis[row, col]

                for m in range(nb_outcomes):
                    cummul += pvals[m]

                    if cummul > unis_n:
                        z[0][row, col] = m
#                         z[0][row, col] = 13
                        break

                # If we reached the end, use the last value.
                # If we have a real distribution [0,1], than this should never
                # happen, right? I got a segmentation fault when removing it.
                # 2014-04-08
                # This might happen due to rounding errors. 2014-05-01
                if z[0][row, col] == -1:
                    z[0][row, col] = nb_outcomes - 1;

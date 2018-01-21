import theano
import numpy

# Caution: Setting this to true prevents a model to be stored correctly. Loading
# is not possible. This is because the Print function is not properly
# serialized.
PRINT_VARS = False
numpy.set_printoptions(edgeitems=500)

def debug_print(var, name):
    """Wrap the given Theano variable into a Print node for debugging.

    If the variable is wrapped into a Print node depends on the state of the
    PRINT_VARS variable above. If it is false, this method just returns the
    original Theano variable.
    The given variable is printed to console whenever it is used in the graph.

    Parameters
    ----------
    var : Theano variable
        variable to be wrapped
    name : str
        name of the variable in the console output

    Returns
    -------
    Theano variable
        wrapped Theano variable

    Example
    -------
    import theano.tensor as T
    d = T.dot(W, x) + b
    d = debug_print(d, 'dot_product')
    """

    if PRINT_VARS is False:
        return var

    return theano.printing.Print(name)(var)

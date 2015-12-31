import ops
import mnist
from train import train

import numpy
import theano
import theano.tensor as T

_params = {}
def param(name, *args, **kwargs):
    """
    A wrapper for `theano.shared` which enables parameter sharing in models.
    
    Creates and returns theano shared variables similarly to `theano.shared`, 
    except if you try to create a param with the same name as a 
    previously-created one, `param(...)` will just return the old one instead of 
    making a new one.

    This constructor also adds a `param` attribute to the shared variables it 
    creates, so that you can easily search a graph for all params.
    """

    if name not in _params:
        kwargs['name'] = name
        param = theano.shared(*args, **kwargs)
        param.param = True
        _params[name] = param
    return _params[name]

def search(node, critereon):
    """
    Traverse the Theano graph starting at `node` and return a list of all nodes
    which match the `critereon` function. When optimizing a cost function, you 
    can use this to get a list of all of the trainable params in the graph, like
    so:

    `swft.search(cost, lambda x: hasattr(x, "param"))`
    """

    def _search(node, critereon, visited):
        if node in visited:
            return []
        visited.add(node)

        results = []
        if isinstance(node, T.Apply):
            for inp in node.inputs:
                results += _search(inp, critereon, visited)
        else: # Variable node
            if critereon(node):
                results.append(node)
            if node.owner is not None:
                results += _search(node.owner, critereon, visited)
        return results

    return _search(node, critereon, set())

def floatX(x):
    """
    Convert `x` to the numpy type specified in `theano.config.floatX`.
    """

    if theano.config.floatX == 'float16':
        return numpy.float16(x)
    elif theano.config.floatX == 'float32':
        return numpy.float32(x)
    else: # Theano's default float type is float64
        return numpy.float64(x)
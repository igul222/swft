import swft
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def Linear(
        name, 
        input_dims, 
        output_dim, 
        inputs,
        biases=True,
        initialization='glorot'
        ):
    """
    Compute a linear transform of one or more inputs, optionally with a bias.

    input_dims: list of ints, or int (if single input); the dimensionality of
                the input(s).
    output_dim: the dimensionality of the output.
    biases:     whether or not to include a bias term.
    inputs:     a theano variable, or list of variables (if multiple inputs);
                the inputs to which to apply the transform.
    initialization: one of `glorot`, `orthogonal`, `("uniform", range)`.
    """

    if not isinstance(input_dims, list):
        input_dims = [input_dims]
        inputs = [inputs]

    terms = []

    for i, (inp, inp_dim) in enumerate(zip(inputs, input_dims)):
        if initialization == 'glorot':
            weight_values = numpy.random.uniform(
                low=-numpy.sqrt(6. / (inp_dim + output_dim)),
                high=numpy.sqrt(6. / (inp_dim + output_dim)),
                size=(inp_dim, output_dim)
            ).astype(theano.config.floatX)
        elif initialization == 'orthogonal':
            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are "
                                       "supported.")
                flat_shape = (shape[0], numpy.prod(shape[1:]))
                a = numpy.random.normal(0.0, 1.0, flat_shape)
                u, _, v = numpy.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype(theano.config.floatX)
            weight_values = sample((inp_dim, output_dim))
        elif initialization[0] == 'uniform':
            weight_values = numpy.random.uniform(
                low=-initialization[1],
                high=initialization[1],
                size=(inp_dim, output_dim)
            ).astype(theano.config.floatX)
        else:
            raise Exception("Invalid initialization!")

        weight = swft.param(
            name + '.W'+str(i),
            weight_values
        )
        terms.append(T.dot(inp, weight))

    if biases:
        terms.append(swft.param(
            name + '.b',
            numpy.zeros((output_dim,), dtype=theano.config.floatX)
        ))

    return reduce(lambda a,b: a+b, terms)

def rectify(x):
    """ReLU nonlinearity: max(0, x)"""
    return (x + abs(x)) / swft.floatX(2.0)

def Embedding(name, n_symbols, output_dim, indices):
    vectors = swft.param(
        name,
        numpy.random.randn(
            n_symbols, 
            output_dim
        ).astype(theano.config.floatX)
    )

    output_shape = [
        indices.shape[i]
        for i in xrange(indices.ndim)
    ] + [output_dim]

    return vectors[indices.flatten()].reshape(output_shape)

def BatchNormalize(name, input_dim, inputs, stepwise=False):
    """
    Batch normalization. By default, normalizes across all but the last axis.
    Set `stepwise` to true if you're batch-norming an RNN and want to normalize
    each timestep separately (e.g. for a language model, where you can't let
    information from step `t+1` leak into step `t`).
    """
    if stepwise:
        means = inputs.mean(axis=1, keepdims=True)
        variances = inputs.var(axis=1, keepdims=True)
    else:
        means = inputs.reshape((-1, input_dim)).mean(axis=0)
        variances = inputs.reshape((-1, input_dim)).var(axis=0)

    beta = swft.param(
        name + '.beta',
        numpy.zeros(input_dim, dtype='float32')
    )

    gamma = swft.param(
        name + '.gamma',
        numpy.ones(input_dim, dtype='float32')
    )

    stdevs = T.sqrt(variances + swft.floatX(1e-4))

    return (inputs - means) * (gamma / stdevs) + beta

def Dropout(p_drop, inputs):
    """
    Drop each input randomly with probability `p_drop`, and scale the remaining
    ones to preserve the overall variance. This op doesn't yet support a
    test-time mode (where all inputs are kept).
    """
    srng = RandomStreams(seed=234)
    scaled_inputs = inputs / swft.floatX(1-p_drop)
    return scaled_inputs * srng.binomial(
        inputs.shape, 
        p=swft.floatX(1-p_drop),
       dtype=theano.config.floatX
    )
import numpy
import theano
import theano.ifelse
import theano.tensor as T
import swft

def RNNStep(name, input_dim, hidden_dim, current_input, last_hidden):
    return T.tanh(
        swft.ops.Linear(
            name, 
            [input_dim, hidden_dim], 
            hidden_dim,
            [current_input, last_hidden]
        )
    )

def Recurrent(name, hidden_dim, step_fn, inputs, reset=None):
    h0 = swft.param(
        name + '.h0',
        numpy.zeros((hidden_dim,), dtype=theano.config.floatX)
    )

    num_batches = inputs.shape[1]
    h0_batched = T.alloc(h0, num_batches, hidden_dim) 

    if reset is not None:
        # The shape of last_hidden doesn't matter right now; we assume
        # it won't be used until we put something proper in it.
        last_hidden = theano.shared(
            numpy.zeros((1,1), dtype=theano.config.floatX),
            name=name+'.last_hidden'
        )
        h0_after_reset = theano.ifelse.ifelse(reset, h0_batched, last_hidden)
    else:
        h0_after_reset = h0_batched

    outputs, _ = theano.scan(
        step_fn,
        sequences=inputs,
        outputs_info=h0_after_reset
    )

    if reset is not None:
        last_hidden.default_update = outputs[-1]

    return outputs

def GRU(name, input_dim, hidden_dim, inputs, reset=None):

    processed_inputs = swft.ops.Linear(
        name+'.Input',
        input_dim,
        3 * hidden_dim,
        inputs,
        biases=False
    )

    processed_inputs = swft.ops.BatchNormalize(
        name+'.BatchNormalize',
        3 * hidden_dim,
        processed_inputs,
        stepwise=True
    )

    def step(current_processed_input, last_hidden):
        gates = T.nnet.sigmoid(
            swft.ops.Linear(
                name+'.Recurrent_Gates', 
                hidden_dim, 
                2 * hidden_dim, 
                last_hidden,
                biases=False
            ) + current_processed_input[:, :2*hidden_dim]
        )

        update = gates[:, :hidden_dim]
        reset  = gates[:, hidden_dim:]

        scaled_hidden = reset * last_hidden

        candidate = T.tanh(
            swft.ops.Linear(
                name+'.Recurrent_Candidate', 
                hidden_dim, 
                hidden_dim, 
                scaled_hidden,
                biases=False,
                initialization='orthogonal'
            ) + current_processed_input[:, 2*hidden_dim:]
        )

        one = swft.floatX(1.0)
        return (update * candidate) + ((one - update) * last_hidden)

    return Recurrent(
        name+'.Recurrent',
        hidden_dim,
        step,
        processed_inputs,
        reset
    )

def flatten3D(inputs):
    return inputs.reshape((
        inputs.shape[0] * inputs.shape[1],
        inputs.shape[2]
    ))
import swft
import numpy
import theano
import theano.tensor as T
import lasagne

import math
import time
import locale

locale.setlocale(locale.LC_ALL, '')

def _print_paramsets_info(costs, paramsets):
    """Print information about the parameters in the given param sets."""

    for cost, params in zip(costs, paramsets):
        params = sorted(params, key=lambda p: p.name)
        shapes = [p.get_value(borrow=True).shape for p in params]
        print "Params for cost {0}:".format(cost.name)
        for param, shape in zip(params, shapes):
            print "\t{0} ({1})".format(
                param.name,
                ",".join([str(x) for x in shape])
            )

        total_param_count = 0
        for shape in shapes:
            param_count = 1
            for dim in shape:
                param_count *= dim
            total_param_count += param_count
        print "Total parameter count: {0}".format(
            locale.format("%d", total_param_count, grouping=True)
        )

def train(
        symbolic_inputs,
        costs,
        train_data,
        dev_data=None,
        test_data=None,
        param_sets=None,
        optimizers=[lasagne.updates.adam],
        print_vars=None,
        epochs=10,
        print_every=10,
        callback=None
    ):
    # TODO write documentation

    if param_sets == None:
        param_sets = [ swft.search(costs[0], lambda x: hasattr(x, 'param')) ]

    assert len(costs)==len(param_sets), "train() needs 1 param set per cost!"

    _print_paramsets_info(costs, param_sets)

    print "Building updates..."

    if print_vars is None:
        print_vars = [c for c in costs]
    for cost in costs:
        print_vars += swft.search(cost, lambda x: hasattr(x, '_print'))
    # Remove duplicate values in print_vars
    print_vars = list(set(print_vars))

    all_updates = []
    for cost, params, optimizer in zip(costs, param_sets, optimizers):
        grads = T.grad(cost, wrt=params)
        # Clip gradients elementwise
        grads = [
            T.clip(g, swft.floatX(-1.0), swft.floatX(1.0))
            for g in grads
        ]

        cost_updates = optimizer(grads, params)
        for k, v in cost_updates.items():
            all_updates.append((k,v))

    print "Compiling train function..."

    train_ = theano.function(
        symbolic_inputs, 
        print_vars,
        updates=all_updates,
        on_unused_input='warn'
    )

    print "Compiling evaluate function..."

    evaluate = theano.function(
        symbolic_inputs, 
        print_vars,
        on_unused_input='warn'
    )

    print "Training!"

    splits = [
        ('train', train_, train_data)
    ]
    if dev_data is not None:
        splits.append(('dev', evaluate, dev_data))
    if test_data is not None:
        splits.append(('test', evaluate, test_data))

    for epoch in xrange(epochs):
        for title, fn, data in splits:

            epoch_totals      = []
            since_last_print  = []
            n_inputs = 0

            for iteration, inputs in enumerate(data(), start=1):
                n_inputs += 1

                start_time = time.time()

                outputs_ = fn(*inputs)

                if iteration == 1:
                    epoch_totals     = [o.copy() for o in outputs_]
                    since_last_print = [o.copy() for o in outputs_]
                else:
                    for i, o in enumerate(outputs_):
                        epoch_totals[i]     += o
                        since_last_print[i] += o

                if iteration % print_every == 0:

                    new_time = time.time()

                    values_to_print = [
                        ('epoch', epoch),
                        ('input', iteration),
                        ('time_per_input', (time.time() - start_time))
                    ]

                    for symbolic, totalval in zip(print_vars, since_last_print):
                        values_to_print.append(
                            (str(symbolic), totalval / print_every)
                        )

                    print "{0}\t".format(title) + "\t".join([
                        "{0}:{1}".format(name, val)
                        for name, val in values_to_print
                    ])

                    last_print_time = new_time

                    for i, t in enumerate(since_last_print):
                        since_last_print[i].fill(0)

            values_to_print = [
                ('epoch', epoch),
                ('n_inputs', n_inputs)
            ]

            for symbolic_var, total_val in zip(print_vars, epoch_totals):
                values_to_print.append(
                    (str(symbolic_var), total_val / n_inputs)
                )

            print "{0} summary\t".format(title) + "\t".join(
                ["{0}:{1}".format(name, val) for name, val in values_to_print]
            )

        if callback:
            callback(epoch)
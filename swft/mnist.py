import numpy
import theano
import theano.tensor as T

import os
import urllib
import gzip
import cPickle as pickle

def mnist_generator(data, batch_size):
    images, targets = data
    images = images.astype(theano.config.floatX)
    targets = targets.astype(theano.config.floatX)
        
    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)
        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)
        for i in xrange(len(image_batches)):
            yield (image_batches[i], target_batches[i])

    return get_epoch

def symbolic_inputs():
    return [T.matrix(), T.vector()]

def load(batch_size):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    return (
        mnist_generator(train_data, batch_size), 
        mnist_generator(dev_data, 5000), 
        mnist_generator(test_data, 5000)
    )
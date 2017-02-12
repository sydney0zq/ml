#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo for CNN.
"""

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

mini_batch_size = 10
training_data, validation_data, test_data = network3.load_data_shared()

"""
net = Network([FullyConnectedLayer(n_in=784, n_out=100), SoftmaxLayer(n_in=100, n_out=10)],
                mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
"""

"""
Best validation accuracy of 97.75% obtained at iteration 114999
Corresponding test accuracy of 97.72%
"""

#using conv layer
"""
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                        filter_shape = (20, 1, 5, 5),
                        poolsize = (2, 2)),
               FullyConnectedLayer(n_in=20*12*12, n_out=100),
               SoftmaxLayer(n_in = 100, n_out = 10)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
"""

"""
Best validation accuracy of 98.83% obtained at iteration 129999
Corresponding test accuracy of 98.75%
"""


#using two conv layers
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                        filter_shape = (20, 1, 5, 5),
                        poolsize = (2, 2)),
               ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                        filter_shape = (40, 20, 5, 5),
                        poolsize = (2, 2)), 
               FullyConnectedLayer(n_in=40*4*4, n_out=100),
               SoftmaxLayer(n_in = 100, n_out = 10)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

exit()

#replace the sigmoid to ReL
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                        filter_shape = (20, 1, 5, 5),
                        poolsize = (2, 2),
                        activation_fn = network3.ReLU),
               ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                        filter_shape = (40, 20, 5, 5),
                        poolsize = (2, 2),
                        activation_fn = network3.ReLU), 
               FullyConnectedLayer(n_in=40*4*4, n_out=100),
               SoftmaxLayer(n_in = 100, n_out = 10)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data, lmbda = 0.1)


#expand the training_data
#$ python expand_mnist.py

expanded_training_data, _, _ = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                        filter_shape = (20, 1, 5, 5),
                        poolsize = (2, 2),
                        activation_fn = network3.ReLU),
               ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                        filter_shape = (40, 20, 5, 5),
                        poolsize = (2, 2),
                        activation_fn = network3.ReLU), 
               FullyConnectedLayer(n_in=40*4*4, n_out=100),
               SoftmaxLayer(n_in = 100, n_out = 10)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)


#add dropout to the last full-connected layer
expanded_training_data, _, _ = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                        filter_shape = (20, 1, 5, 5),
                        poolsize = (2, 2),
                        activation_fn = network3.ReLU),
               ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                        filter_shape = (40, 20, 5, 5),
                        poolsize = (2, 2),
                        activation_fn = network3.ReLU), 
               FullyConnectedLayer(n_in=40*4*4, n_out=1000, activation_fn = network3.ReLU, p_dropout = 0.5),
               FullyConnectedLayer(n_in=1000, n_out=1000, activation_fn = network3.ReLU, p_dropout = 0.5),
               SoftmaxLayer(n_in = 1000, n_out = 100, p_dropout = 0.5)], 
               mini_batch_size)

net.SGD(training_data, 40, mini_batch_size, 0.03, validation_data, test_data)











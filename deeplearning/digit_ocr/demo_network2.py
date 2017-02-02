#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
A comparsion between the old approach and new approach.


本章 CH2 从以下方面做了提高：

cost 函数：cross entropy
regularization L1 and L2
softmax layer
初始化：1/sqrt(n_in)
"""


import mnist_loader
import network2


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


"""
从这个例子中看到新的初始化方法只是增加了学习的速率，最终表现是一样的，但在一些神经网络中
新的初始化权重的方法会提高最终的 accuracy
"""

#Using the old approach, standard Gauss distribution
net = network2.Network([784, 30, 10], cost = network2.CrossEntropyCost)
net.large_weight_initalizer()
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, evaluation_data = validation_data, monitor_evaluation_accuracy = True)

#Using the new approach, N(0, 1/sqrt(n_in))
net = network2.Network([784, 30, 10], cost = network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, evaluation_data = validation_data, monitor_evaluation_accuracy = True)










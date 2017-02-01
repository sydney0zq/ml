#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-

import mnist_loader
import network2

#:training_data: 是一个List, 50000 个 tuples, 784*1 和 10*1 的vector
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10], cost = network2.CrossEntropyCost)                                  
net.large_weight_initializer()  #和之前初始化是一样的, 专门标记一下是因为之后我们会用其他方法初始化
net.SGD(training_data, 30, 10, 0.5, evaluation_data = test_data, monitor_evaluation_accuracy=True)             


"""
Epoch 0 training complete
Accuracy on evaluation data: 9110 / 10000

Epoch 1 training complete
Accuracy on evaluation data: 9257 / 10000

Epoch 2 training complete
Accuracy on evaluation data: 9373 / 10000

...

Epoch 10 training complete
Accuracy on evaluation data: 9485 / 10000

Epoch 11 training complete
Accuracy on evaluation data: 9482 / 10000

Epoch 12 training complete
Accuracy on evaluation data: 9489 / 10000

Epoch 13 training complete
Accuracy on evaluation data: 9496 / 10000

Epoch 14 training complete
Accuracy on evaluation data: 9507 / 10000

Epoch 15 training complete
Accuracy on evaluation data: 9514 / 10000

Epoch 16 training complete
Accuracy on evaluation data: 9436 / 10000

Epoch 17 training complete
Accuracy on evaluation data: 9509 / 10000

Epoch 18 training complete
Accuracy on evaluation data: 9529 / 10000

Epoch 19 training complete
Accuracy on evaluation data: 9537 / 10000

Epoch 20 training complete
Accuracy on evaluation data: 9480 / 10000

Epoch 21 training complete
Accuracy on evaluation data: 9515 / 10000

Epoch 22 training complete
Accuracy on evaluation data: 9510 / 10000

Epoch 23 training complete
Accuracy on evaluation data: 9543 / 10000

Epoch 24 training complete
Accuracy on evaluation data: 9537 / 10000

Epoch 25 training complete
Accuracy on evaluation data: 9549 / 10000

Epoch 26 training complete
Accuracy on evaluation data: 9534 / 10000

Epoch 27 training complete
Accuracy on evaluation data: 9565 / 10000

Epoch 28 training complete
Accuracy on evaluation data: 9552 / 10000

Epoch 29 training complete
Accuracy on evaluation data: 9536 / 10000
"""

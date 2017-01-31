#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-


import mnist_loader

#:training_data: 是一个List, 50000 个 tuples, 784*1 和 10*1 的vector
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print "training_data type:", type(training_data)
print "training_data len:", len(training_data) 
print "training_data[0][0].shape:", training_data[0][0].shape
print "training_data[0][1].shape:", training_data[0][1].shape

print "validation_data len:", len(validation_data)
print "test_data len:", len(test_data)

import network
#net = network.Network([784, 30, 10])

#2th: 30 epochs
#3th: mini_batch_size每次用10个来学习
#4th: 学习率
#net = network.Network([784, 100, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

"""
Epoch 8: 9426 / 10000
Epoch 9: 9451 / 10000
Epoch 10: 9451 / 10000
Epoch 11: 9427 / 10000
Epoch 12: 9399 / 10000
Epoch 13: 9459 / 10000
Epoch 14: 9464 / 10000
Epoch 15: 9449 / 10000
Epoch 16: 9475 / 10000
Epoch 17: 9471 / 10000
Epoch 18: 9458 / 10000
Epoch 19: 9464 / 10000
Epoch 20: 9475 / 10000
Epoch 21: 9487 / 10000
Epoch 22: 9495 / 10000
Epoch 23: 9482 / 10000
Epoch 24: 9481 / 10000
Epoch 25: 9499 / 10000
"""

#net.SGD(training_data, 10, 10, 3.0, test_data=test_data)

"""
Epoch 0: 4242 / 10000
Epoch 1: 5856 / 10000
Epoch 2: 7807 / 10000
Epoch 3: 8479 / 10000
Epoch 4: 8473 / 10000
Epoch 5: 8567 / 10000
Epoch 6: 8572 / 10000
Epoch 7: 8588 / 10000
Epoch 8: 8585 / 10000
Epoch 9: 8600 / 10000
"""

#net = network.Network([784, 30, 10])
#net.SGD(training_data, 30, 10, 0.001, test_data=test_data)

"""
Epoch 0: 1370 / 10000
Epoch 1: 1320 / 10000
Epoch 2: 1211 / 10000
Epoch 3: 1266 / 10000
Epoch 4: 1654 / 10000
Epoch 5: 1833 / 10000
Epoch 6: 1942 / 10000
Epoch 7: 2027 / 10000
Epoch 8: 2093 / 10000
Epoch 9: 2172 / 10000
Epoch 10: 2233 / 10000
Epoch 11: 2288 / 10000
Epoch 12: 2340 / 10000
Epoch 13: 2415 / 10000
Epoch 14: 2467 / 10000
Epoch 15: 2530 / 10000
Epoch 16: 2567 / 10000
Epoch 17: 2611 / 10000
"""

#只有输入输出层
net = network.Network([784, 10])
net.SGD(training_data, 30, 10, 3, test_data=test_data)











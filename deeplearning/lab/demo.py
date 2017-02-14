#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-


import svhn_loader

#:training_data: 是一个List, 50000 个 tuples, 784*1 和 10*1 的vector
training_data, test_data = svhn_loader.load_data_wrapper()

print "training_data type:", type(training_data)
print "training_data len:", len(training_data) 
print "training_data[0][0].shape:", training_data[0][0].shape
print "training_data[0][1].shape:", training_data[0][1].shape

print "test_data len:", len(test_data)

import network3
net = network3.Network([3072, 100, 10])

#2th: 30 epochs
#3th: mini_batch_size每次用10个来学习
#4th: 学习率
#net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)










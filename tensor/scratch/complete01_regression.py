#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
https://www.tensorflow.org/get_started/get_started
"""


import tensorflow as tf

import numpy as np

#model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
#model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

#loss
loss = tf.reduce_sum(tf.square(linear_model - y))       #sum of the squares
#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#training_data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

#training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

#evaluate training accuarcy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print ("*" * 100)
print ("W: %s; b: %s; loss:%s." % (curr_W, curr_b, curr_loss))











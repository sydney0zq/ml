#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
You might think of TensorFlow Core programs as consisting of two discrete sections:

    Building the computational graph.
    Running the computational graph.
<https://www.tensorflow.org/get_started/get_started>
"""

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)    #also tf.float32 implictly
print(node1, node2)
#output
#(<tf.Tensor 'Const:0' shape=() dtype=float32>, <tf.Tensor 'Const_1:0' shape=() dtype=float32>)

"""
Notice that printing the nodes does not output the values 3.0 and 4.0 as you might expect. Instead, they are nodes that, when evaluated, would produce 3.0 and 4.0, respectively. To actually evaluate the nodes, we must run the computational graph within a session. A session encapsulates the control and state of the TensorFlow runtime.
"""

sess = tf.Session()
print (sess.run([node1, node2]))


print("Now run node3...")

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))
#output
#('node3: ', <tf.Tensor 'Add:0' shape=() dtype=float32>)
#('sess.run(node3): ', 7.0)

print ("*" * 30)
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b          #provide a shortcut for tf.add(a, b)
print (sess.run(adder_node, {a:1, b:4.5}))
print (sess.run(adder_node, {a:[1, 3], b:[2, 4]}))

add_and_triple = adder_node * 3
print (sess.run(add_and_triple, {a:3, b:4.5}))

"""
Constants are initialized when you call tf.constant, and their value can never change. By contrast, variables are not initialized when you call tf.Variable. To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:
"""
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

print (sess.run(linear_model, {x:[1, 2, 3, 4]}))

print ("*" * 40)
print ("Calc the loss value...")
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print (sess.run(loss, {x: [1,2,3,4], y: [0, -1, -2, -3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
"""
We guessed the "perfect" values of W and b, but the whole point of machine learning is to find the correct model parameters automatically. We will show how to accomplish this in the next section.
"""
print (sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


































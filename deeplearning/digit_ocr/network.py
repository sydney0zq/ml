#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
network.py
------------------

A module to implement the SGD learning algorithm for a feedfoward NN. Gradient
are calculated using backprogation.

Note the code is simple, easily readable, and easily modifiable. It is not
optimized, and omits many desirable features.
"""

import random
import numpy as np


def sigmoid(z):
    #note that the input z is a vector or np.array, np automatically
    #applies the function sigmoid elementwise in vectorized form
    return 1.0 / (1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1-sigmoid(z))



class Network(object):

    def __init__(self, sizes):
        """
        The lists `sizes`（每层神经元的个数） contains the number of neurons in the respective
        layers of the network. i.e. if the list was [2, 3, 1] then it would be 3
        layer network. The biases and weights for the network are inited randomly,
        using a Gaussian distribution with mean 0, and varience 1. NOTE that the
        first layer is assumed to be an input layer, and by convention we wont set
        any biases for those neurons, since biases are only ever used in computing
        the output from later layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        #random.randn(y, 1) 随机从正态分布（均值 0,  方差 1) 中生成
        #sizes = [2,3,1]
        #test = [np.random.randn(y, 1) for y in sizes[1:]]
        #>>> test
        #[array([[ 1.94011169],
        #        [ 0.80066664],
        #        [-1.41180998]]), array([[-0.93776222]])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #net.weights[1] 存储连接第二层和第三层的权重 (python 从 0 开始存放）
        #>>> sizes
        #[1, 2, 3, 4, 5, 6]
        #>>> for x, y in zip(sizes[:-1], sizes[1:]):
        #         print(x, y)
        #    1 2
        #    2 3
        #    3 4
        #    4 5
        #    5 6
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        """
        Return the output of the network if `a` is input.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, eta,
                    test_data=None):
        """
        Train the NN using mini-batch stochastic gradient descent.
        The `training_data` is a list of tuples `(x, y)` representing the
        training inputs and the desired outputs. The other non-optional
        parameters are self-explanatory. If `test_data` is provided then
        network will be evaluated against the test data after each epoch,
        and partial progress printed out. This useful for tracking pregress,
        but slow things down substantially.
        :param eta: the learning rate.

        :training_data: 是一个 list, 包括了很多 tuple, 每一个 tuple 包含了一个 x 和 y,
        x 即为 numpy.array 的数据类型。y 即为 label。
        :epochs: 是根据我们的具体数据结构设置的，让结果收敛即可。
        :mini_batch_size: 每一小块有多少个实例。
        :eta: 学习速率。
        :test_data:
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            #shuffle 洗牌，接受一个 list, 把这个 list 中的元素随机打乱
            random.shuffle(training_data)
            #抽取数据。假如 mini_batches_size 是 100, 挑法是 [0, n) 中间每次
            #隔 mini_batch_size。[0,99), [100,199) ... 形成多个 mini_batch
            mini_batches = [training_data[k:k+mini_batch_size]
                                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                #update_mini_batch 是最重要的，更新 weight 和 bias
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                #每完成一轮，看看准确性。
                print("Epoch {0}: {1} / {2}".format(
                        j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
    

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying
        GD using backpropagation to a single mini batch. The
        "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate.
        最关键的一步。backpropagation在计算cost函数的时候计算对
        b和w的偏导数分别是多少的一种方法。

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #x是一幅图片, y是标签。
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #如果不清楚这部分的话, 画图显示一下矩阵的形式就可以明白
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #Equation 20 and 21 in the pdf document
        self.weights = [w - (eta/len(mini_batch))*nw 
                            for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - (eta/len(mini_batch))*nb
                            for b, nb in zip(self.biases, nabla_b)]

    
    def backprop(self, x, y):
        """
        Return a tuple `nabla_b, nabla_w` representing the gradient for the 
        cost function C_x. `nabla_b` and `nabla_w` are layer-by-layer lists 
        of numpy arrays, similar to `self.biases` and `self.weights`.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x] #list to store all the activations, layer by layer
        zs = []     #list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the Neural network 
        outputs the correct result. Note that the NN's output is assumed 
        to be the index of whichever neuron in the final layer has the 
        highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    
    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives \partial C_x / \partial a
        for the output activations.
        """
        return (output_activations - y)






















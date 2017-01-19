#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
The first application code using python in nerual network.

Documents:
About non-linear transformation function

sigmoid function(S curve) is used as **activation function**:
    1. 双曲函数(tanh)
    2. 逻辑函数(logistic function)

<https://en.wikipedia.org/wiki/Sigmoid_function>
<https://zh.wikipedia.org/wiki/双曲函数>
<https://zh.wikipedia.org/wiki/邏輯函數>

<https://rolisz.ro/2013/04/18/neural-networks-in-python/>
"""

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    #__init__ 构造函数
    #若调用不指定activation, 那么默认为tanh
    def __init__(self, layers, activation = 'tanh'):
        """
        :param layers: A list containing the number of units in each layer
        Should be at least two values
        :param activation: The activation function to be used, can be 'logistic' or 'tanh'
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1): 
            #对于某一层, 前后两个连线都需要产生权重
            #<https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.random.random.html>
            #Three-by-two array of random numbers from [-5, 0):
            #>>> 5 * np.random.random_sample((3, 2)) - 5
            #array([[-3.99149989, -0.52338984],
            #       [-2.99091858, -0.79479508],
            #       [-1.23204345, -1.75224494]])
            #numpy.random.random(size=None)
            #Return random floats in the half-open interval [0.0, 1.0).
            #这里是很重要的一点: 如果想搞清楚必须画图演示一下才能清晰过程和结果
            self.weights.append((2 * np.random.random((layers[i-1]+1, layers[i]+1))-1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i]+1, layers[i+1])) - 1) * 0.25)
        #print (">>>Weights list is \n", self.weights, "\n>>>EOF")
    """
    :params epochs: Meaning that we will sample to update. One back and forth means
    one `epochs`. 10000 limits the maximum.
    """
    def fit(self, X, y, learning_rate = 0.2, epochs = 10000):
        #Make sure X is at least two dimensions, one for instance, the other for the features
        X = np.atleast_2d(X)
        #shape[0] --> ROW; shape[1] --> COLUMN
        #the `+1` means the init bias
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        #from column 0 to the last-1 column
        temp[:, 0:-1] = X       # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        #key steps
        for k in range(epochs):
            #random a row to update(Return random integers from low (inclusive) to high (exclusive))
            i = np.random.randint(X.shape[0])
            a = [X[i]]          #assgin the random row to a

            for l in range(len(self.weights)):      #going to forward network, for each layer
                #`np.dot(a[l], self.weights[l]` complete the <sum yi * wij>
                #`self.activation` complete the activatation
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            #at the end we arrived at the last layer
            #compute the <T_j - O_j>, cool coding
            error = y[i] - a[-1]
            #compute the Err_j
            deltas = [error * self.activation_deriv(a[-1])]

            #starting back propagation
            #from 0 --> len(a)-2, every time minus 1
            #len(a)-2, len(a)-3, len(a)-4, ..., 0
            for l in range(len(a)-2, 0, -1):        #we need to begin at the second to last layer
                deltas.append( deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]) )
            deltas.reverse()        #because we goes back inversely
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                #delta weight += learning_rate * Err_j * O_i
                self.weights[i] += learning_rate * layer.T.dot(delta)
    
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0 : -1] = x
        a = temp
        for l in range(0, len(self.weights)):
            #We donot need to save all layers
            a = self.activation(np.dot(a, self.weights[l]))
        return a









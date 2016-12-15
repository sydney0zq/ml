#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-12-09 root <root@VM-17-202-debian>

import numpy as np 

#Strategy #2: Random local search

def CIFAR10_loss_fun(W):
    return L(X_train, Y_train, W)

def eval_numberical_gradient(f, x):
    """
    A naive implementation of numberical gradient of f at x
    -f should be a function that takes a single argument
    -x is the point (numpy array) to evaluate the gradient at
    """
    fx = f(x)   #evaluate function value at original point
    grad = np.zeros(x.shape)
    h = 0.00001

    #iterate over all indexes in x
    #np.nditer: It inter as follows:
    #------------->
    #...
    #You should know that it.multi.index is the index
    #of the matrix.And do not forget to interate
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        #evaluate function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h   #increment by h
        fxh = f(x)              #evaluate f(x+h)
        x[ix] = old_value       #restore to previous value!!
        #compute the partial derivative
        grad[ix] = (fxh - fx) / h   #the slope
        it.iternext()           #step to next dimension
    return grad
"""
#generate random starting W
W = np.random.randn(10, 3073) * 0.001
bestloss = float("inf")
for i in xrange(1000):
    step_size = 0.0001
    Wtry = W + np.random.randn(10, 3073) * step_size
    loss = L(Xtr_cols, Ytr, Wtry)
    if loss < bestloss:
        W = Wtry
        bestloss = loss
    print "inter %d loss is %f" % (i, bestloss)
"""


W = np.random.rand(10, 1037) * 0.001    #return weight vector
#The gradient tells us the slope of the loss function along every function
df = eval_numberical_gradient(CIFAR10_loss_fun, W)   #get the gradient

loss_original = CIFAR10_loss_fun(W)
print 'original loss: %f' % (loss_original, )

#lets see the effect of multiple step sizes
for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
    step_size = 10 ** step_size_log
    w_new = W - step_size_log
    W_new = W - step_size * df
    loss_new = CIFAR10_loss_fun(W_new)
    print 'for step size %f new loss: %f' % (step_size, loss_new)








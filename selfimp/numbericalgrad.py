#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-12-09 root <root@VM-17-202-debian>

import numpy as np
from linearclassification import L_i
from data_utils import load_CIFAR10


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
    #------------->
    #You should know that it.multi_index is the index
    #of the matrix. And do not forget to interate
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


def CIFAR_loss_fun(W):
    return L_i(X_train, Y_train, W)


X_train, Y_train, X_te, Y_te = load_CIFAR10("../CIFAR")


W = np.random.rand(10, 3073) * 0.001        #Random weight
df = eval_numberical_gradient(CIFAR_loss_fun, W)    #Get the grad





#print f(np.array([[1,2]]))
print df









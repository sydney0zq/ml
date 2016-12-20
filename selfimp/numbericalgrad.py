#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-12-09 root <root@VM-17-202-debian>

import numpy as np
from linearclassification import *
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
    #use L_i_vectorized function, after it I will use
    #pure L without iteration
    for i in range(0, Xtr.shape[0]):
        y_pos = Ytr[i]
        #This chooses each column of training data and
        #full W to compute scores
        #wrong: loss_i = L_i_vectorized(Xtr_col[i], y_pos, W[:, i])
        loss_i[0, i] = L_i_vectorized(Xtr_col[:, i], y_pos, W)        
    return loss_i

Xtr, Ytr, Xte, Yte = load_CIFAR10("../CIFAR")
Xtr_col = Xtr.reshape(32*32*3, Xtr.shape[0])
ones_row = np.ones((1, Xtr.shape[0]))
Xtr_col = np.append(Xtr_col, ones_row, axis = 0)

loss_i = np.zeros((1, Xtr.shape[0]))

W = np.random.rand(10, 3073) * 0.001        #Random weight
df = eval_numberical_gradient(CIFAR_loss_fun, W)    #Get the grad

print df













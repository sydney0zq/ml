#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-12-05 root <root@VM-17-202-debian>

import numpy as np

#Here is the loss function (without regularization) implemented in Python, in both unvectorized and half- vectorized form:

def L_i(x, y, W):
    """
    unvectorized version. Compute the multiclass SVM loss for
    a single example (x, y)
    - x is a column vector representing an image
    e.g. 3073*1 in CIFAR-10 with an appended bias dimension
    in the 3073-rd position(i.e base trick ch3linear classification)
    - y is an integer giving index of correct class
    e.g. between 0 and 9 in CIFAR-10
    - W is the weight matrix
    e.g. 10*3073 in CIFAR-10
    Note that each row is a classifier
    """
    delta = 1.0     #see notes about delta later in this section
    scores = W.dot(x)   #x is the column vector, which is a picture
    correct_class_score = scores[y] #tells which is the right class
    D = W.shape[0]  #number of classes, e.g. 10
    loss_i = 0.0    #L_i in the ch3 note
    for j in xrange(D): #interate over all wrong classes
        if j == y:
            #skip for the true class to only loop over incorrect classes
            continue
        loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i

def L_i_vectorized(x, y, W):
    """
    A faster half-vectorized implementation. Half-vectorized refers
    to the fact for a single example the implementation contains
    no for loops, but there is still one loop over the examples
    (outside this function)
    """
    delta = 1.0;
    scores = W.dot(x);
    #compare the margins for all classes in one vector operation
    margins = np.maximum(0, scores - scores[y] + delta)
    #on y-th position scores[y] - scores[y] canceled and gave delta
    #we want to ignore the y-th position and only consider matgin 
    #on max wrong class
    margins[y] = 0;
    loss_i = np.sum(margins)
    return loss_i

def L(X, y, W):
    """
    fully-vectorized implementation:
    -X holds all the training examples as columns
    e.g. 3073 * 50000 in CIFAR-10
    -y is array of integers specifying correct class
    e.g. 50000-D array
    -W are weight matrix
    e.g. 10 * 3073
    """
    #evaluate loss over all examples in X without using any loops
    #left as exercise to reader in the assignment

























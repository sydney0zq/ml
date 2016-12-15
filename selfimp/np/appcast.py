#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-15 root <root@VM-17-202-debian>

import numpy as np

#computer outer product of vectors
v = np.array([1,2,3])   #shape (3,)
w = np.array([4,5])     #shape (2,)

# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:

print np.reshape(v, (3,1))
print np.reshape(v, (3,1)) * w

x = np.array([[1,2,3], [4,5,6]])
print x + v

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
print (x.T + w).T
# Another solution is to reshape w to be a row vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same output.
print x + np.reshape(w, (2,1))

print x
print x*2





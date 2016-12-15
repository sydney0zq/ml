#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-15 root <root@VM-17-202-debian>

import numpy as np

x = np.array([[1,2], [3,4]], dtype = np.float64)
y = np.array([[5,6], [7,8]], dtype = np.float64)

# elementwise sum, both produce the array
print x + y
print np.add(x, y)

#elementwise difference, produce the array
print x - y
print np.subtract(x, y)

#elementwise product; both produce the array
print x * y 
print np.multiply(x, y)

#elementwise division,both produce the array
print x / y
print np.divide(x, y)

#elementwise square root produce the array
print np.sqrt(x)


print "*" * 40

#Note that unlike MATLAB, * is elementwise multiplication, not matrix multiplication. We instead use the dot function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices. dot is available both as a function in the numpy module and as an instance method of array objects:


#*是元素逐个相乘，而不是矩阵乘法。在Numpy中使用dot来进行矩阵乘法
v = np.array([9, 10])
w = np.array([11,12])

#inner product of vectors, both produce 219
print v.dot(w)
print np.dot(v, w)

#matrix / vector product; both produce the rank 1 array
print x.dot(v)
print np.dot(x, v)

#matrix/ matrix product; both produce the rank 2 array

print x.dot(y)
print np.dot(x,y)












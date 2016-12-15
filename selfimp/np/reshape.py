#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-15 root <root@VM-17-202-debian>

import numpy as np

#Apart from computing mathematical functions using arrays, we frequently need to reshape or otherwise manipulate data in arrays. The simplest example of this type of operation is transposing a matrix; to transpose a matrix, simply use the T attribute of an array object:

x = np.array([[1,2], [3,4]])
print x
print x.T

print "*" * 30
#note that taking the transpose of a rank 1 array does nothing
v = np.array([1,2,3])
print v 
print v.T

print "*" * 30
#note the difference
v = np.array([[1,2,3]])
print v 
print v.T
print "*" * 30

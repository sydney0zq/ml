#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016 11 14 root <root@VM-17-202-debian>

#Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.

import numpy as np

#create a rank 1 array
a = np.array([1,2,3])
#prints <type 'numpy.ndarray'>
print type(a)

#print (3,)
print a.shape
#print 1 2 3
print a[0], a[1], a[2]
a[0] = 5
#print 5 2 3
print a

#create a rank 2 array
b = np.array([[1,2,3],[4,5,6]])
#print (2, 3)
print b.shape
#print 1 2 4
print b[0,0], b[0,1], b[1,0]

###########################

print "Now start to create some new matrix"

#create an array of all zeros
a = np.zeros((2,2))
print a

b = np.ones((1,2))
print b

#create a constant array
c = np.full((2,2), 7)
print c

#create a 2*2 identity matrix
d = np.eye(2)
print d

#create an array filled with random values
e = np.random.random((2,2))
print e


















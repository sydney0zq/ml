#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016 11 14 root <root@VM-17-202-debian>

import numpy as np

#Integer array indexing: When you index into numpy arrays using slicing, the resulting array view will always be a subarray of the original array. In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array.

a = np.array([[1,2], [3,4], [5,6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and 

#well it means (0,0), (1,1), (2,0)
print a[[0,1,2], [0,1,0]]
#prints > [1 4 5]

# The above example of integer array indexing is equivalent to this:
print np.array([a[0,0],a[1,1],a[2,0]])
#prints > [1 4 5]

# When using integer array indexing, you can reuse the same
# element from the source array:
print a[[0,0],[1,1]]

print np.array([a[0,1], a[0,1]])

print "*" * 30

a = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])

print a

#create an array of indices
b = np.array([0,2,0,1])
print "b is: " 
print b

#select one element from each row of a using the indices in b 
print a[np.arange(4), b]
#prints > [ 1  6  7 11]

a[np.arange(4), b] += 11

print a











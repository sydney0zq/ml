#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016 11 14 root <root@VM-17-202-debian>

#Numpy offers several ways to index into arrays.

#Slicing: Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array:

import numpy as np

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

#use slicing to pull out the subarray of the first 2 rows
#and columns 1 and 2; b is the following array of the shape(2,2)
b = a[:2, 1:3]

print "array a is"
print a

print "array b is"
print b

print "*" * 30
#print '2'
print a[0, 1]
#b[0,0] is the same piece of the same data as a[0,1]
b[0,0] = 77
print a[0,1]

print "*" * 30
# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:

#rank 1 view of the second row of a
row_r1 = a[1, :]
#rank 2 view of the second row of a
row_r2 = a[1:2, :]
print  row_r1, row_r1.shape
print  row_r2, row_r2.shape

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print col_r1, col_r1.shape
print col_r2, col_r2.shape















#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-15 root <root@VM-17-202-debian>

#Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing arithmetic operations. Frequently we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.

import numpy as np

#we will add the vector v to each row of the matrix x 
#storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1, 0, 1])

#create an empty matrix with the same shape as x
y = np.empty_like(x)
print y

#add the vector v to each row of the matrix x with an explict loop
for i in range(4):
    y[i, :] = x[i, :] + v
print "*"  * 20
print y

#This works; however when the matrix x is very large, computing an explicit loop in Python could be slow. Note that adding the vector v to each row of the matrix x is equivalent to forming a matrix vv by stacking multiple copies of v vertically, then performing elementwise summation of x and vv. We could implement this approach like this:

#stack 4 copies of v on top of each other
vv = np.tile(v, (4,1))
print vv

y = x + vv
print y


#Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v. Consider this version, using broadcasting:
#add v to each row of x using broadcasting
y = x + v
print y

#The line y = x + v works even though x has shape (4, 3) and v has shape (3,) due to broadcasting; this line works as if v actually had shape (4, 3), where each row was a copy of v, and the sum was performed elementwise.

#**********************************
#Broadcasting two arrays together follows these rules:

#If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
#1. The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
#2. The arrays can be broadcast together if they are compatible in all dimensions.
#3. After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
#4. In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension

#https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html












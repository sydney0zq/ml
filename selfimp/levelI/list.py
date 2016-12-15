#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Copyright © 2016 root <root@VM-17-202-debian>

#create a list
xs = [3, 1, 2]

print xs, xs[2]
print xs[-1]
xs[2] = 'foo'
print xs
xs.append('bar')
print xs

print xs.pop(), xs

nums = [0,1,2,3]
squares = []
for x in nums:
    squares.append(x**2)
print squares

even_squares = [x**2 for x in nums if x > 0]
print even_squares



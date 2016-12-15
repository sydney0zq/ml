#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016 root <root@VM-17-202-debian>

#A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot. 

d = {(x, x+1): x for x in range(10)}
print d
t = (5, 6)
print type(t)
#prints > type is tuple
print d[t]
print d[(1,2)]





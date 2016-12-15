#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016 root <root@VM-17-202-debian>

#A set is an unordered collection of distinct elements.


animals = {'cat', 'dog'}
#check if an element is in a set, print true or false
print 'cat' in animals
print 'fish' in animals

animals.add('fish')
print 'fish' in animals

print len(animals)
animals.add('cat')
print len(animals)
animals.remove('cat')
print len(animals)

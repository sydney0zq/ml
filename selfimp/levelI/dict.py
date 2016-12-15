#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016 root <root@VM-17-202-debian>

d = {'cat': 'cute', 'dog': 'furry'}

print d['cat']

print 'cat' in d

d['fish'] = 'wet'

print d['fish']

# Get an element with a default; prints "N/A"
print d.get('monkey', 'N/A')
print d.get('fish', 'N/A')

del d['fish']
print d.get('fish', 'N/A')

d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print 'A %s has %d legs...' % (animal, legs)
# Prints "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs"

print "*" * 10
for animal,legs in d.iteritems():
    print 'A %s has %d legs' % (animal, legs)

nums = [0,1,2,3,4]
print type(nums)
even_num_to_square = {x: x**2 for x in nums if x%2 ==0}
print even_num_to_square
print type(even_num_to_square)

animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx+1, animal)



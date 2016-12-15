#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016 root <root@VM-17-202-debian>

animals = ['cat', 'dog', 'monkey']

for animal in animals:
    print animal
#access to the index of each element within the body of a loop,
#use the built-in enumerate function:

for idx, animal in enumerate(animals):
    print "#%d: %s" % (idx+1, animal)


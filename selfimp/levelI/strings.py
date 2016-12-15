#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Copyright © 2016 root <root@VM-17-202-debian>

hello = "hello"
world = "world"

print "%s %s %s" % (hello, world, 12)

s = hello

print s.capitalize()    #Hello
print s.upper()         #HELLO
print s.rjust(7)        #   hello
print s.center(7)       #  hello  
print s.replace('l','sssb')       

print '   world '.strip()   #strip leading and trailing whitespace









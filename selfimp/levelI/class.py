#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016 11 14 root <root@VM-17-202-debian>

class Greeter(object):
    #Constructor
    def __init__(self, name):
    #Create an instance variable
        self.name = name

    #Instance method
    def greet(self, loud=False):
        if loud:
            print 'Hello, %s' % self.name.upper()
        else:
            print 'Hello, %s' % self.name

#Construct an instance of Greeter class
g = Greeter('Fred')
#call an instance method
g.greet()
g.greet(loud=True)


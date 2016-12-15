#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016 11 14 root <root@VM-17-202-debian>

def sign(x):
    if x>0:
        return 'positive'
    if x<0:
        return 'negative'
    else:
        return 'zero'

for x in [-1,0,1]:
    print sign(x)



def hello(name, loud=False):
    if loud:
        print 'HELLO, %s!' % name.upper()
    else:
        print 'Hello, %s' % name

hello('Bob')
hello('Fred', loud= True)

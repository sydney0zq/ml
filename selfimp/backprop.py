#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-12-26 root <root@VMdebian>

#set some inputs

x = -2; y = 5; z = -4

#perform the forward pass
q = x + y
f = q * z


dfdz = q
dfdq = z

dfdx = 1.0 * dfdq
dfdy = 1.0 * dfdq



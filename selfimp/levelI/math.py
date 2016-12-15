#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016 root <root@VM-17-202-debian>

from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print nums
#prints >  set([0, 1, 2, 3, 4, 5])



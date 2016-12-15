#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-12-05 root <root@VM-17-202-debian>

c = get_config()
c.IPKernelApp.pylab = 'inline'
c.NotebookApp.password = u'sha1:b20c91a65caa:31f068ab25fd53f1f6397084e508663eeadb5fa7'

c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8080



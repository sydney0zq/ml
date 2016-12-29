#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-12-05 root <root@VM-17-202-debian>

#Forward/Backward API

class ComputationalGraph(object):
    # This is the API class for computing graph backprop
    def forward(inputs):
        #1. [pass inputs to input gates...]
        #2. forward the computational graph:
        for gate in self.graph.node_topologically_sorted():
            gate.forward()
        return loss #The final gate in the graph outputs the loss
    def backward():
        for gate in reversed(self.gate.node_topologically_sorted()):
            gate.backward() #little piece of backprop(chain rule applied)
        return inputs_gradients

class MultipyGate(object):
    """
        Attention x, y, z are scalars
    """
    def forward(x, y):
        z = x * y
        self.x = x      #Must keep these around
        self.y = y
    def backward(dz):
        dx = self.y * dz
        dy = self.x * dx
        return [dx, dy]

















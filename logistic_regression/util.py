#!/usr/bin/python

import sys
import os
import numpy

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

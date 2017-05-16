#!/usr/bin/python
# -*- coding=utf-8 -*-
"""
MNIST DNN demo by tobycc
"""
import tensorflow as tf

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

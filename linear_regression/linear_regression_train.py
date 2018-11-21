#!/usr/bin/python
import sys
import os
import random
import math

if len(sys.argv) != 5:
    print "ERROR:python linear_regression_train.py train_data_file learning_rate max_iteration convergence"
    sys.exit(-1)

input_file_name = sys.argv[1]
learning_rate = float(sys.argv[2])
max_iter = int(sys.argv[3])
conv = float(sys.argv[4])

input_file = open(input_file_name, "r")
samples = input_file.readlines()
input_file.close()

sample_num = len(samples)
feature_num = len(samples[0].split(",")) - 2

print "INFO:%d samples, %d features" % (sample_num, feature_num)

# init model
model = []
for i in xrange(0, feature_num + 1):
    model.append(random.uniform(-100, 100))
print "DEBUG:init model with"
print model

cost_function_value_list = []

# train iter
for i in xrange(0, max_iter):
    print "INFO:iter %d" % i
    # calculate cost function and calculate each parameter gradient
    cost_function_value = 0.0
    model_gradient = [0.0] * (feature_num + 1)
    for sample in samples:
        attr = sample.strip().split(',')
        label_real = float(attr[1])
        label_predict = model[-1]
        # cost function
        for feature_index in xrange(0, feature_num):
            label_predict = label_predict + model[feature_index] * float(attr[feature_index + 2])
        # print "DEBUG: sample %s label %f predict %f predict - real %f" % (attr[0], label_real, label_predict, label_predict - label_real)
        cost_function_value = cost_function_value + math.pow((label_real - label_predict), 2)

        # parameter gradient
        for feature_index in xrange(0, feature_num):
            model_gradient[feature_index] = model_gradient[feature_index] + (label_predict - label_real) * float(attr[feature_index + 2])
        model_gradient[-1] = model_gradient[-1] + label_predict - label_real
    
    cost_function_value = cost_function_value / (2 * sample_num)
    cost_function_value_list.append(cost_function_value)
    for feature_index in xrange(0, feature_num + 1):
        model_gradient[feature_index] = model_gradient[feature_index] / sample_num

    print "INFO:cost function value %f" % cost_function_value
    print "DEBUG:model parameter gradient"
    print model_gradient

    # if already convergence
    if i >= 5:
        cost_function_value_descent_percent = (cost_function_value_list[-5] - cost_function_value_list[-1]) / \
                                              cost_function_value_list[-5]
        print "DEBUG:cost function value descent percent = %f" % cost_function_value_descent_percent
        if cost_function_value_descent_percent >= 0 and \
           cost_function_value_descent_percent <= conv:
            print "INFO:ealy stop at iter %d" % i
            break

    # update model parameter by gradient descent
    for feature_index in xrange(0, feature_num + 1):
        model[feature_index] = model[feature_index] - learning_rate * model_gradient[feature_index]
    print "DEBUG:model after iter %d" % i
    print model

print "INFO:model result %s" % ",".join(map(lambda x:str(x), model)) 

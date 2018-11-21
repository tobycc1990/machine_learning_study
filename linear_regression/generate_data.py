#!/usr/bin/python

import sys
import os
import random

if len(sys.argv) != 5:
    print "ERROR:python generate_data.py feature_dim train_num test_num output_index"
    sys.exit(-1)

feature_dim = int(sys.argv[1])
train_num = int(sys.argv[2])
test_num = int(sys.argv[3])
output_index = sys.argv[4]

parameters = []

for i in xrange(0, feature_dim+1):
    parameters.append(round(random.uniform(-100, 100), 2))

print "PARAMETERS:"
print parameters

parameter_output = open("%s_result" % output_index, "w")
parameter_output.write(",".join(map(lambda x:str(x), parameters)))
parameter_output.close()

train_data = []
test_data = []

print "TRAINDATA:"
train_output = open("%s_traindata" % output_index, "w")
for i in xrange(0, train_num):
    sample = []
    result = 0
    for j in xrange(0, feature_dim):
        weight = round(random.uniform(-1, 1), 2)
        result = result + parameters[j] * weight
        sample.append(str(weight))
    result = result + parameters[-1]
    offset = random.uniform(0.95, 1.05)
    result = round(result * offset, 2)
    line = "%d,%.2f,%s" % (i, result, ",".join(sample))
    print line
    train_output.write(line)
    train_output.write("\n")
train_output.close()

print "TESTDATA:"
test_output = open("%s_testdata" % output_index, "w")
for i in xrange(0, test_num):
    sample = []
    result = 0
    for j in xrange(0, feature_dim):
        weight = round(random.uniform(-1, 1), 2)
        result = result + parameters[j] * weight
        sample.append(str(weight))
    result = result + parameters[-1]
    offset = random.uniform(0.95, 1.05)
    result = round(result * offset, 2)
    line = "%d,%.2f,%s" % (i, result, ",".join(sample))
    print line
    test_output.write(line)
    test_output.write("\n")

test_output.close()

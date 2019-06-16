import os
import argparse
import numpy as np
import glob
import scipy.misc
import ast
import traceback
import sys
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--data-path', dest = 'data_path', default = '../data_unbalanced_new/')
parser.add_argument('--out-data-path', dest = 'out_data_path', default = "out/")
parser.add_argument('--test-chunk', dest = 'test_chunk', type = int, default = 1)

args = parser.parse_args()

train_data = np.genfromtxt("features.csv", delimiter = ',')

train_cv_data = train_data[train_data[:, 1] != args.test_chunk]    
test_cv_data = train_data[train_data[:, 1] == args.test_chunk]        
test_amount = len(test_cv_data)
train_amount = len(train_cv_data)    

out_path_train_class_0 = args.out_data_path + '/train/0'
out_path_train_class_1 = args.out_data_path + '/train/1'
out_path_test_class_0 = args.out_data_path + '/test/0'
out_path_test_class_1 = args.out_data_path + '/test/1'

if not os.path.exists(out_path_train_class_0):
    os.makedirs(out_path_train_class_0)
if not os.path.exists(out_path_train_class_1):
    os.makedirs(out_path_train_class_1)
if not os.path.exists(out_path_test_class_0):
    os.makedirs(out_path_test_class_0)
if not os.path.exists(out_path_test_class_1):
    os.makedirs(out_path_test_class_1)
    
for d in test_cv_data:
    id = int(d[0])
    class_id = int(d[2])
    fname = 'data' + str(id).zfill(9) + 'r.png'
    from_path = args.data_path + '/' + fname
    to_path = args.out_data_path + '/test/' + str(class_id) + '/' + fname
    shutil.copyfile(from_path, to_path)
    
for d in train_cv_data:
    id = int(d[0])
    class_id = int(d[2])
    fname = 'data' + str(id).zfill(9) + 'r.png'
    from_path = args.data_path + '/' + fname
    to_path = args.out_data_path + '/train/' + str(class_id) + '/' + fname
    shutil.copyfile(from_path, to_path)    
    
with open(args.out_data_path + '/sizes.txt', 'w') as f:
    f.write("%d\n" % len(train_cv_data))
    f.write("%d\n" % len(test_cv_data))

with open(args.out_data_path + '/classes.txt', 'w') as f:
    f.write("%d\n" % len(train_cv_data[train_cv_data[:, 2] == 0]))
    f.write("%d\n" % len(train_cv_data[train_cv_data[:, 2] != 0]))
    


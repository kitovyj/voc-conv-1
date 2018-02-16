import sys
import os
import numpy as np
import argparse
import time
import re
import scipy.misc
import random
import tensorflow as tf
import math
import string
import glob
import traceback

parser = argparse.ArgumentParser()

parser.add_argument('--cv-folder', dest = 'cv_folder', default = './train/cv/', help = 'the cv folder to get the data from')

args = parser.parse_args()

out_file = "DNN_Learning_Weights.csv"

f = open(out_file, 'a+')
csv_line = '';
csv_line += '"Batches",'
csv_line += '"Performance",'
csv_line += '"Weight change(abs)"'
print(csv_line, file = f)
f.close()                

fn = os.path.join(args.cv_folder, "pretrained.tfevents")
    
accuracies = []
weight_change = []
batches = []
current_batch = 0

for e in tf.train.summary_iterator(fn):
    for v in e.summary.value:
        if v.tag == "accuracy":
            accuracies.append(v.simple_value)
            batches.append(current_batch)
            current_batch += 51
        elif  v.tag == "wca1":
            weight_change.append(v.simple_value)

            

fn = os.path.join(args.cv_folder, "pretrained-1.tfevents")

# 5202
# skip duplicate

current_batch -= 51
del batches[-1]
del weight_change[-1]
del accuracies[-1]

for e in tf.train.summary_iterator(fn):
    for v in e.summary.value:
        if v.tag == "accuracy":
            accuracies.append(v.simple_value)
            batches.append(current_batch)
            current_batch += 34
        elif  v.tag == "wca1":
            weight_change.append(v.simple_value)

            
f = open(out_file, 'a+')

for i in range(len(batches)):

    csv_line = '';

    csv_line += str(batches[i]) + ','
    csv_line += str(accuracies[i]) + ','
    csv_line += str(weight_change[i])

    print(csv_line, file = f)

f.close()                
                

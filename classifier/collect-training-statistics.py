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

parser.add_argument('--cv-folder', dest = 'cv_folder', default = './train/cv-raw-to-gender/6-cv-layers-3-stages-proper-shuffling/0/', help = 'the cv folder to get the data from')

args = parser.parse_args()

out_file = "DNN_Learning_Weights.csv"

f = open(out_file, 'a+')
csv_line = '';
csv_line += '"Batches",'
csv_line += '"Performance",'
csv_line += '"Weight change(abs)"'
print(csv_line, file = f)
f.close()                
    
accuracies = []
weight_change = [0]
batches = []

file_names = ["pretrained.tfevents", "pretrained-1.tfevents", "pretrained-2.tfevents"]

for fn in file_names:

    fn = os.path.join(args.cv_folder, fn)

    for e in tf.train.summary_iterator(fn):
        for v in e.summary.value:
            if v.tag == "accuracy":
                accuracies.append(v.simple_value)
            elif  v.tag == "wca1":
                weight_change.append(v.simple_value)
            elif v.tag == "batch_number":
                batches.append(v.simple_value)
                
                        
f = open(out_file, 'a+')

for i in range(len(batches)):

    csv_line = '';

    csv_line += str(int(batches[i])) + ','
    csv_line += str(accuracies[i]) + ','
    csv_line += str(weight_change[i])

    print(csv_line, file = f)

f.close()                
                

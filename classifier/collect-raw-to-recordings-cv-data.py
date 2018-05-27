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

folders = sorted(glob.glob(args.cv_folder + '/*/'))

out_file = "DNN_Raw2Recording_Real.csv"

total_recordings = 17
total_female = 9
total_male = 8

f = open(out_file, 'w+')
csv_line = '';

for i in range(total_female):
    csv_line += '"Female' + str(i + 1) + '",'

for i in range(total_male):
    csv_line += '"Male' + str(i + 1) + '",'

csv_line += '"Overall"'
    
print(csv_line, file = f)
f.close()                

for s in folders:

    fn = os.path.join(s, "pretrained-1.tfevents")
    
    accuracy = 0
    accuracies = [0 for i in range(total_male + total_female)]
    
    for e in tf.train.summary_iterator(fn):
        for v in e.summary.value:
            if v.tag == "accuracy":
                accuracy = v.simple_value    
            elif re.match('accuracy_[0-9]+', v.tag):
                #print(v.tag)
                split = v.tag.split('_')
                num = int(split[1][0:])                
                accuracies[num - 1] = v.simple_value
                
                
    f = open(out_file, 'a+')

    csv_line = '';

    for i in range(total_female + total_male):
        csv_line += str(accuracies[i]) + ','

    csv_line += str(accuracy)
    
    print(csv_line, file = f)

    f.close()                

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

parser.add_argument('--summary-file', dest = 'summary_file', default = '.\train\cv-raw-to-gender\6-cv-layers-3-stages-proper-shuffling\0\pretrained-2.tfevents', help = 'the summary file` folder to get the data from')
parser.add_argument('--feature', dest = 'feature', type = int, default = 0, help = 'feature number')
parser.add_argument('--initial', action='store_true', dest='take_initial', help = 'if we have to store initial weights values')

args = parser.parse_args()

out_file = "Kernel_"
if args.take_initial:
    out_file = out_file + "Initial"
else:
    out_file = out_file + "Trained"

out_file = out_file + "_" + str(args.feature + 1) + ".csv"

f = open(out_file, 'a+')
f.close()                

postfix = ""
if args.take_initial:
    postfix = "-original"

def tensor_summary_value_to_numpy(value):
    fb = np.frombuffer(value.tensor.tensor_content, dtype = np.float32)

    shape = []
    for d in value.tensor.tensor_shape.dim:
        shape.append(d.size)
    #fb.reshape(reversed(shape))
    fb = fb.reshape(shape)
    return fb
    
for e in tf.train.summary_iterator(args.summary_file):

    do_break = False

    for v in e.summary.value:

        if v.node_name == "":
            v.node_name = v.tag    
        
        if re.match('c[0-9]+-weights' + postfix, v.node_name) :
           split = v.node_name.split('-')
           num = int(split[0][1:])
           print("loading convolutional layer " + str(num) + " weights")
           
           if num != 1:
               continue
           
           w = tensor_summary_value_to_numpy(v)

           do_break = True
 
           break

        '''   
        if (v.tag is not None) and (len(v.tag) > 0):
            print(v.tag + ' loaded')
        else:
        '''
        #print(v.node_name + ' loaded')
 
    if do_break:
        break
       
f = open(out_file, 'a+')

print(w.shape)

w = w[:, :, :, args.feature]

print(w.shape)

width = w.shape[1]
height = w.shape[0]

for y in range(height):
    csv_line = '';
    for x in range(width):
        v = w[y, x, 0]
        csv_line += str(v)
        if x != width - 1:
            csv_line += ','
    print(csv_line, file = f)

f.close()                
                

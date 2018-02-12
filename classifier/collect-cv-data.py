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

out_file = "DNN_Full_Real.csv"

f = open(out_file, 'a+')
csv_line = '';
csv_line += '"Female",'
csv_line += '"Male",'
csv_line += '"Overall"'
print(csv_line, file = f)
f.close()                

for s in folders:

    #fn = os.path.join(s, "2", "pretrained-1.tfevents")
    fn = os.path.join(s, "pretrained-1.tfevents")
    
    accuracy = 0
    gender_accuracies = [0, 0]
    
    for e in tf.train.summary_iterator(fn):
        for v in e.summary.value:
            if v.tag == "accuracy":
                accuracy = v.simple_value
            elif  v.tag == "accuracy_1":
                gender_accuracies[0] = v.simple_value                
            elif  v.tag == "accuracy_2":
                gender_accuracies[1] = v.simple_value
                
                
    f = open(out_file, 'a+')

    csv_line = '';

    csv_line += str(gender_accuracies[0]) + ','
    csv_line += str(gender_accuracies[1]) + ','
    csv_line += str(accuracy)
    
    print(csv_line, file = f)

    f.close()                
                
    '''

    #sz = sys.getsizeof(e)
        #print(str(sz))
        print(e)

        for v in e.summary.value:
            
            
        
            if v.tag.startswith('conv'):
                content = v.image.encoded_image_string
                fname = 'conv' + str(i).zfill(9) + '.png'
                with open(fname, 'wb') as f:
                    f.write(content)    
                    i = i + 1    
    '''
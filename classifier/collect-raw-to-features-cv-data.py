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

out_file = "DNN_Raw2Features_Real.csv"
incline_file = "DNN_Raw2FeaturesIncline_Real.csv"
peaks_file = "DNN_Raw2FeaturesPeaks_Real.csv"
breaks_file = "DNN_Raw2FeaturesBreaks_Real.csv"

total_incline = 3
total_peaks = 4
total_breaks = 4

f = open(out_file, 'w+')
csv_line = '';
csv_line += '"Incline",'
csv_line += '"Peaks",'
csv_line += '"Breaks",'
csv_line += '"Broadband",'
csv_line += '"Tremolo",'
csv_line += '"Complex"'
print(csv_line, file = f)
f.close()                

f = open(incline_file, 'w+')
csv_line = '';
csv_line += '"Incline = -1",'
csv_line += '"Incline = 0",'
csv_line += '"Incline = 1",'
csv_line += '"Overall"'
print(csv_line, file = f)
f.close()                

f = open(peaks_file, 'w+')
csv_line = '';
csv_line += '"No peaks",'
csv_line += '"1 peak",'
csv_line += '"2 peaks",'
csv_line += '"3 peaks",'
csv_line += '"Overall"'
print(csv_line, file = f)
f.close()                

f = open(breaks_file, 'w+')
csv_line = '';
csv_line += '"No breaks",'
csv_line += '"1 break",'
csv_line += '"2 breaks",'
csv_line += '"3 breaks",'
csv_line += '"Overall"'
print(csv_line, file = f)
f.close()                

for s in folders:

    fn = os.path.join(s, "pretrained-1.tfevents")
    
    incline_accuracies = [0] * total_incline
    incline_accuracy = 0
    peaks_accuracies = [0] * total_peaks
    peaks_accuracy = 0
    breaks_accuracies = [0] * total_breaks
    breaks_accuracy = 0

    broadband_accuracy = 0
    tremolo_accuracy = 0
    complex_accuracy = 0    
    
    for e in tf.train.summary_iterator(fn):
        for v in e.summary.value:
            if v.tag == "incline_accuracy":
                incline_accuracy = v.simple_value
            elif v.tag == "peaks_accuracy":
                peaks_accuracy = v.simple_value
            elif v.tag == "breaks_accuracy":
                breaks_accuracy = v.simple_value
            elif v.tag == "broadband_accuracy":
                broadband_accuracy = v.simple_value
            elif v.tag == "tremolo_accuracy":
                tremolo_accuracy = v.simple_value
            elif v.tag == "complex_accuracy":
                complex_accuracy = v.simple_value               
            elif re.match('incline_accuracy_[0-9]+', v.tag):
                #print(v.tag)
                split = v.tag.split('_')
                num = int(split[2][0:])                
                incline_accuracies[num] = v.simple_value
            elif re.match('peaks_accuracy_[0-9]+', v.tag):
                #print(v.tag)
                split = v.tag.split('_')
                num = int(split[2][0:])                
                peaks_accuracies[num] = v.simple_value
            elif re.match('breaks_accuracy_[0-9]+', v.tag):
                #print(v.tag)
                split = v.tag.split('_')
                num = int(split[2][0:])                
                breaks_accuracies[num] = v.simple_value
                
                
    f = open(out_file, 'a+')

    csv_line = '';

    csv_line += str(incline_accuracy) + ','
    csv_line += str(peaks_accuracy) + ','
    csv_line += str(breaks_accuracy) + ','
    csv_line += str(broadband_accuracy) + ','
    csv_line += str(tremolo_accuracy) + ','
    csv_line += str(complex_accuracy)

    print(csv_line, file = f)

    f.close()                

    f = open(incline_file, 'a+')
    csv_line = '';
    for a in incline_accuracies:
        csv_line += str(a) + ','
    csv_line += str(incline_accuracy)
    print(csv_line, file = f)
    f.close()                

    f = open(peaks_file, 'a+')
    csv_line = '';
    for a in peaks_accuracies:
        csv_line += str(a) + ','
    csv_line += str(peaks_accuracy)
    print(csv_line, file = f)
    f.close()                

    f = open(breaks_file, 'a+')
    csv_line = '';
    for a in breaks_accuracies:
        csv_line += str(a) + ','
    csv_line += str(breaks_accuracy)
    print(csv_line, file = f)
    f.close()                

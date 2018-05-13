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

parser.add_argument('--gender-folder', dest = 'gender_folder', default = './raw-data-gender/', help = 'fender folder')
parser.add_argument('--permuted-folder', dest = 'permuted_folder', default = './data_unbalanced_new/', help = 'permuted folder')

args = parser.parse_args()

print('reading permuted...')

permuted_files = glob.glob(args.permuted_folder + '/*.csv')

permuted = {}

for f in permuted_files:
    
    #print(os.path.basename(f))
    
    '''
    if os.path.basename(f).find("r") != -1:
        continue
    '''
        
    #print(os.path.basename(f))
        
    fh = open(f, 'rb')
    arr = fh.read()
    fh.close()
    permuted[arr] = f

print("total permuted:", len(permuted))
    
print('mapping female...')
    
female_files = glob.glob(args.gender_folder + '/1/*.csv')

female_2_permuted = []

def fn2id(fn):
    t = fn[0:4]
    id = int(fn[4:(4+9)])        
    return id
    
for f in female_files:

    '''
    if os.path.basename(f).find("r") != -1:
        continue

    '''
    
    fh = open(f, 'rb')
    arr = fh.read()
    fh.close()
    
    if arr in permuted:
        pfn = permuted[arr]
        pfn = os.path.basename(pfn)
        fn = os.path.basename(f)                
        female_2_permuted.append((fn, pfn))
        permuted[arr] = "already.matched"
    else:
        print("not found!")
    
        
np.savetxt('female-2-permuted.csv', female_2_permuted, delimiter = ',', fmt = "%s")        
    
print('mapping male...')
    
male_files = glob.glob(args.gender_folder + '/2/*.csv')

male_2_permuted = []

for f in male_files:

    '''
    if os.path.basename(f).find("r") != -1:
        continue
    '''
        
    fh = open(f, 'rb')
    arr = fh.read()
    fh.close()
    
    if arr in permuted:
        pfn = permuted[arr]
        pfn = os.path.basename(pfn)
        fn = os.path.basename(f)
        male_2_permuted.append((fn, pfn))
        permuted[arr] = "already.matched"
    else:
        print("not found!")
        
np.savetxt('male-2-permuted.csv', male_2_permuted, delimiter = ',', fmt = "%s")        
    

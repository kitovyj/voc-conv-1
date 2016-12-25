#!/bin/bash
for i in 1024 512 256 64 32 16 8 4 2 1
do
   python3 -u ./voc-conv.py --fc-size $i
done
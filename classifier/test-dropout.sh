#!/bin/bash
for i in 0.0 0.25 0.5 0.75
do
   python3 -u ./voc-conv.py --fc-size 128 --dropout $i
done
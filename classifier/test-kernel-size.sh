#!/bin/bash
for i in {2..10}
do
 
    python3 -u ./voc-conv-real.py --data-path ./vocs_data2/ --train-amount 12454 --epochs 300 --initial-weights-seed 9 --fc-num 3 --fc-sizes 512 256 128 --kernel-size $i

done

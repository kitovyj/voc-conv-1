#!/bin/bash
python -u ./voc-conv-real.py --epochs 500 --initial-weights-seed 1 --fc-num 2 --fc-sizes 512 512
python -u ./voc-conv-real.py --epochs 500 --initial-weights-seed 1 --fc-num 2 --fc-sizes 256 256
python -u ./voc-conv-real.py --epochs 500 -initial-weights-seed 1 --fc-num 3 --fc-sizes 128 128 128
python -u ./voc-conv-real.py --epochs 500 -initial-weights-seed 1 --fc-num 3 --fc-sizes 512 256 128
python -u ./voc-conv-real.py --epochs 500 -initial-weights-seed 1 --fc-num 4 --fc-sizes 512 256 128 64
python -u ./voc-conv-real.py --epochs 500 -initial-weights-seed 1 --fc-num 5 --fc-sizes 512 256 128 64 32

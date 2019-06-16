SET W=16

SET /A F1=16*9+6
SET /A F2=16*8+8
SET /A F3=16*13+8
SET /A F4=16*10+13

for %%f in (%F1%, %F2%, %F3%, %F4%) do (

python collect-weights-data.py --summary-file d:\dnn\voc-conv-1\classifier\train\cv-raw-to-gender\6-cv-layers-3-stages-proper-shuffling\0\pretrained.tfevents --feature %%f --initial
python collect-weights-data.py --summary-file d:\dnn\voc-conv-1\classifier\train\cv-raw-to-gender\6-cv-layers-3-stages-proper-shuffling\0\pretrained-2.tfevents --feature %%f

)

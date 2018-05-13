SET W=16

SET /A F1=16*4+2
SET /A F2=16*5+7
SET /A F3=16*6+4
SET /A F4=16*10+6
SET /A F5=16*11+4
SET /A F6=16*13+1
SET /A F7=16*14+1
SET /A F8=16*8+12
SET /A F9=16*7+8
SET /A F10=16*3+1

for %%f in (%F1%, %F2%, %F3%, %F4%, %F5%, %F6%, %F7%, %F8%, %F9%, %F10%) do (

python collect-weights-data.py --summary-file d:\dnn\voc-conv-1\classifier\train\cv-weights-change\0\pretrained.tfevents --feature %%f --initial
python collect-weights-data.py --summary-file d:\dnn\voc-conv-1\classifier\train\cv-weights-change\0\pretrained-1.tfevents --feature %%f

)

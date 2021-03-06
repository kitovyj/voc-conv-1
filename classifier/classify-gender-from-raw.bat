SET EPOCHS0=40
SET EPOCHS1=25
SET SR=100

FOR /L %%C IN (0,1,9) DO (

    python -u ./train-gender-from-raw.py --epochs %EPOCHS1% --kernel-sizes 10 5 3 3 3 3 --features 256 256 256 256 256 256 --max-pooling 1 1 1 1 1 1 --strides 2 2 1 1 1 1 --fc-num 3 --fc-sizes 120 120 120 --dropout 0.7 --regularization 0.0000 --learning-rate 0.00001 --batch-normalization --summary-records %SR% --test-chunk %%C --summary-file train\cv\%%C\pretrained-2.tfevents --classify
    
)

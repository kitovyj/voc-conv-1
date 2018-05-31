SET EPOCHS0=40
SET EPOCHS1=25
SET SR=100

MKDIR train
MKDIR train\cv

FOR /L %%C IN (0,1,9) DO (

    MKDIR train\cv\%%C

    python -u ./train-features-from-raw.py --epochs %EPOCHS0% --kernel-sizes 10 5 3 3 3 3 --features 256 256 256 256 256 256 --max-pooling 1 1 1 1 1 1 --strides 2 2 1 1 1 1 --fc-num 3 --fc-sizes 120 120 120 --dropout 0.0 --regularization 0.0000 --learning-rate 0.001 --batch-normalization --summary-records %SR% --test-chunk %%C
    REN train\*.tfevents* pretrained.tfevents
    MOVE train\pretrained.tfevents train\cv\%%C\

    python -u ./train-features-from-raw.py --epochs %EPOCHS1% --kernel-sizes 10 5 3 3 3 3 --features 256 256 256 256 256 256 --max-pooling 1 1 1 1 1 1 --strides 2 2 1 1 1 1 --fc-num 3 --fc-sizes 120 120 120 --dropout 0.5 --regularization 0.0000 --learning-rate 0.0001 --batch-normalization --summary-records %SR% --test-chunk %%C --summary-file train\cv\%%C\pretrained.tfevents
    REN train\*.tfevents* pretrained-1.tfevents
    MOVE train\pretrained-1.tfevents train\cv\%%C\

)
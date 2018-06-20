SET EPOCHS0=700
SET EPOCHS1=400
SET SR=100

MKDIR train
MKDIR train\cv

FOR /L %%C IN (0,1,9) DO (

    MKDIR train\cv\%%C

    rem --batch-normalization 
    
    python -u ./train-gender-from-features.py --epochs %EPOCHS0% --fc-num 4 --fc-sizes 70 70 70 70 --dropout 0.0 --regularization 0.0000 --learning-rate 0.001 --summary-records %SR% --test-chunk %%C --batch-normalization
    REN train\*.tfevents* pretrained.tfevents
    MOVE train\pretrained.tfevents train\cv\%%C\

    python -u ./train-gender-from-features.py --epochs %EPOCHS0% --fc-num 4 --fc-sizes 70 70 70 70 --dropout 0.5 --regularization 0.0000 --learning-rate 0.0001 --summary-records %SR% --test-chunk %%C --batch-normalization --summary-file train\cv\%%C\pretrained.tfevents
    REN train\*.tfevents* pretrained-1.tfevents
    MOVE train\pretrained-1.tfevents train\cv\%%C\

    rem python -u ./train-gender-from-features.py --epochs %EPOCHS0% --fc-num 4 --fc-sizes 70 70 70 70 --dropout 0.75 --regularization 0.0000 --learning-rate 0.00002 --summary-records %SR% --test-chunk %%C --batch-normalization --summary-file train\cv\%%C\pretrained-1.tfevents
    rem REN train\*.tfevents* pretrained-2.tfevents
    rem MOVE train\pretrained-2.tfevents train\cv\%%C\    
)
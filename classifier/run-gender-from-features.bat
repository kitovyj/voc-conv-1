SET EPOCHS0=2000
SET EPOCHS1=70
SET SR=100

MKDIR train
MKDIR train\cv

FOR /L %%C IN (0,1,0) DO (

    MKDIR train\cv\%%C

    rem --batch-normalization 
    
    python -u ./train-gender-from-features.py --epochs %EPOCHS0% --fc-num 4 --fc-sizes 50 50 50 50 --dropout 0.0 --regularization 0.0000 --learning-rate 0.00005 --summary-records %SR% --test-chunk %%C
    REN train\*.tfevents* pretrained.tfevents
    MOVE train\pretrained.tfevents train\cv\%%C\

rem    python -u ./train-gender-from-features.py --epochs %EPOCHS1% --fc-num 4 --fc-sizes 50 50 50 50 --dropout 0.5 --regularization 0.0000 --learning-rate 0.0001 --summary-records %SR% --test-chunk %%C --summary-file train\cv\%%C\pretrained.tfevents
rem    REN train\*.tfevents* pretrained-1.tfevents
rem    MOVE train\pretrained-1.tfevents train\cv\%%C\
    
)
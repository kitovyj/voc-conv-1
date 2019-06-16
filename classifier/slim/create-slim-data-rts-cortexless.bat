FOR /L %%C IN (0,1,9) DO (


    MKDIR data-c-rts\%%C
    MKDIR data-c-rts\%%C\train
    MKDIR data-c-rts\%%C\test

   rem  python -u ./prepare-pics-random-test-set-cortexless.py --out-data-path data-c-rts\%%C\ --test-chunk %%C
    python -u ./create_tfrecord.py --dataset_dir data-c-rts\%%C\train\ --test_dataset_dir data-c-rts\%%C\test\   

)
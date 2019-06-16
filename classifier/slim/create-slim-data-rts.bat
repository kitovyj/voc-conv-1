FOR /L %%C IN (0,1,9) DO (


    MKDIR data-rts\%%C
    MKDIR data-rts\%%C\train
    MKDIR data-rts\%%C\test

    python -u ./prepare-pics-random-test-set.py --out-data-path data-rts\%%C\ --test-chunk %%C
    python -u ./create_tfrecord.py --dataset_dir data-rts\%%C\train\ --test_dataset_dir data-rts\%%C\test\   

)
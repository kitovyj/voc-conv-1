FOR /L %%C IN (1,1,18) DO (


    MKDIR data\%%C
    MKDIR data\%%C\train
    MKDIR data\%%C\test

    python -u ./prepare-pics.py --out-data-path data\%%C\ --test-chunk %%C
    python -u ./create_tfrecord.py --dataset_dir data\%%C\train\ --test_dataset_dir data\%%C\test\   

)
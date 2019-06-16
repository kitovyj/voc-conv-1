FOR /L %%C IN (0,1,0) DO (


    MKDIR trained
    MKDIR trained\%%C
    MKDIR trained\%%C\logits    

    python -u ./train_image_classifier.py --train_dir trained\%%C\ --max_number_of_steps 90000 --cv_chunk %%C

    rem python -u ./train_image_classifier.py --checkpoint_exclude_scopes "InceptionV4/Logits,InceptionV4/AuxLogits" --trainable_scopes "InceptionV4/Logits,InceptionV4/AuxLogits" --train_dir trained\%%C\logits --max_number_of_steps 20000 --cv_chunk %%C

    rem python -u ./train_image_classifier.py --checkpoint_path trained\%%C\logits --train_dir trained\%%C\ --max_number_of_steps 40000 --cv_chunk %%C

)
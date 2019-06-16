rem TIMEOUT /T 3600

FOR /L %%C IN (0,1,0) DO (


    MKDIR trained-nn
    MKDIR trained-nn\%%C
    MKDIR trained-nn\%%C\logits    

    rem python -u ./train_image_classifier_nasnet.py --train_dir trained\%%C\ --max_number_of_steps 120000 --cv_chunk %%C

    rem python -u ./train_image_classifier_nasnet.py --checkpoint_path "./tmp\checkpoints\pnasnet\model.ckpt" --checkpoint_exclude_scopes "final_layer, aux_7" --trainable_scopes "final_layer, aux_7" --train_dir trained-nn\%%C\logits --max_number_of_steps 40000 --cv_chunk %%C

    python -u ./train_image_classifier_nasnet.py --checkpoint_path trained-nn\%%C\logits --train_dir trained-nn\%%C\ --max_number_of_steps 120000 --cv_chunk %%C

)
FOR /L %%C IN (0,1,0) DO (


    python -u ./train_image_classifier.py --checkpoint_path trained\%%C\logits --train_dir trained\%%C\ --max_number_of_steps 160000 --cv_chunk %%C

)
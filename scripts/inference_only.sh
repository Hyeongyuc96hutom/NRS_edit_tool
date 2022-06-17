: << "END"  
python inference_only.py \
    --fold '1' \
    --trial 1 \
    --model "mobilenetv3_large_100" \
    --pretrained \
    --use_lightning_style_save \
    --max_epoch 50 \
    --batch_size 256 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --cuda_list '6' \
    --random_seed 3829 \
    --IB_ratio 2 \
    --stage 'general_train' \
    --inference_fold '1' \
    --experiments_sheet_dir '/OOB_RECOG/results-mobilenet-random-ratio' \
    --save_path '/OOB_RECOG/logs/mobilenet-random-ratio' \
    --restore_path '/OOB_RECOG/logs/mobilenet-random-ratio-trial:1-fold:1/TB_log/version_1'

python inference_only.py \
    --fold '1' \
    --trial 1 \
    --model "mobilenetv3_large_100" \
    --pretrained \
    --use_lightning_style_save \
    --max_epoch 50 \
    --batch_size 256 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --cuda_list '6' \
    --random_seed 3829 \
    --IB_ratio 3 \
    --stage 'general_train' \
    --inference_fold '1' \
    --experiments_sheet_dir '/OOB_RECOG/results-mobilenet-random-ratio' \
    --save_path '/OOB_RECOG/logs/mobilenet-random-ratio' \
    --restore_path '/OOB_RECOG/logs/mobilenet-random-ratio-trial:1-fold:1/TB_log/version_2'
END

python inference_only.py \
    --fold '1' \
    --trial 1 \
    --model "mobilenetv3_large_100" \
    --pretrained \
    --use_lightning_style_save \
    --max_epoch 50 \
    --batch_size 256 \
    --lr_scheduler "step_lr" \
    --lr_scheduler_step 5 \
    --lr_scheduler_factor 0.9 \
    --cuda_list '6' \
    --random_seed 3829 \
    --IB_ratio 5 \
    --stage 'general_train' \
    --inference_fold 'free' \
    --experiments_sheet_dir '/OOB_RECOG/results-mobilenet-random-ratio' \
    --save_path '/OOB_RECOG/logs/mobilenet-random-ratio' \
    --restore_path '/OOB_RECOG/logs/mobilenet-random-ratio-trial:1-fold:1/TB_log/version_3'

# python inference_only.py \
#     --fold '1' \
#     --trial 1 \
#     --model "mobilenetv3_large_100" \
#     --pretrained \
#     --use_lightning_style_save \
#     --max_epoch 50 \
#     --batch_size 256 \
#     --lr_scheduler "step_lr" \
#     --lr_scheduler_step 5 \
#     --lr_scheduler_factor 0.9 \
#     --cuda_list '6' \
#     --random_seed 3829 \
#     --IB_ratio 7 \
#     --stage 'general_train' \
#     --inference_fold '1' \
#     --experiments_sheet_dir '/OOB_RECOG/results-mobilenet-random-ratio' \
#     --save_path '/OOB_RECOG/logs/mobilenet-random-ratio' \
#     --restore_path '/OOB_RECOG/logs/mobilenet-random-ratio-trial:1-fold:1/TB_log/version_4'

# python inference_only.py \
#     --fold '1' \
#     --trial 1 \
#     --model "mobilenetv3_large_100" \
#     --pretrained \
#     --use_lightning_style_save \
#     --max_epoch 50 \
#     --batch_size 256 \
#     --lr_scheduler "step_lr" \
#     --lr_scheduler_step 5 \
#     --lr_scheduler_factor 0.9 \
#     --cuda_list '6' \
#     --random_seed 3829 \
#     --IB_ratio 10 \
#     --stage 'general_train' \
#     --inference_fold '1' \
#     --experiments_sheet_dir '/OOB_RECOG/results-mobilenet-random-ratio' \
#     --save_path '/OOB_RECOG/logs/mobilenet-random-ratio' \
#     --restore_path '/OOB_RECOG/logs/mobilenet-random-ratio-trial:1-fold:1/TB_log/version_5'

# : << "END"  
# python inference_only.py \
#     --fold '1' \
#     --trial 1 \
#     --model "mobilenetv3_large_100" \
#     --pretrained \
#     --use_lightning_style_save \
#     --max_epoch 50 \
#     --batch_size 256 \
#     --lr_scheduler "step_lr" \
#     --lr_scheduler_step 5 \
#     --lr_scheduler_factor 0.9 \
#     --cuda_list '6' \
#     --random_seed 3829 \
#     --IB_ratio 10 \
#     --stage 'general_train' \
#     --inference_fold 'free' \
#     --experiments_sheet_dir '/OOB_RECOG/results-temp' \
#     --save_path '/OOB_RECOG/logs/mobilenet-random-ratio' \
#     --restore_path '/OOB_RECOG/logs/mobilenet-random-ratio-trial:1-fold:1/TB_log/version_5'
# END
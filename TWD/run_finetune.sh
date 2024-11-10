CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29500 \
    train.py \
    --model_name_or_path "google/flan-t5-large" \
    --output_dir "question_generation/FT-lora/google-flan-t5-large/lr1e-4-r64" \
    --do_train True \
    --task_name question_generation \
    --max_source_length 512 \
    --max_target_length 128 \
    --per_device_train_batch_size 8  \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 10 \
    --preprocessing_num_workers 4 \
    --fp16 False \
    --seed 42 \
    --predict_with_generate False \
    --prediction_loss_only True \
    --generation_num_beams 4 \
    --label_smoothing_factor 0.15 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --logging_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --use_lora True \
    --lora_rank 64 \
    --lora_alpha 64 \
    # --lr_scheduler_type "cosine"
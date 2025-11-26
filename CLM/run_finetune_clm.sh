CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file "configs/zero3_config.yaml" --main_process_port 29501 --num_processes 4 \
    train_clm.py \
    --model_name_or_path "google/gemma-2-9b-it" \
    --model_type "gemma" \
    --output_dir "question_generation/FT/gemma-2-9b/ff/fp16/e7/lr2e-5-hl-eos" \
    --overwrite_output_dir \
    --do_train True \
    --task_name question_generation \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1  \
    --per_device_eval_batch_size 4 \
    --attn_implementation "eager" \
    --num_train_epochs 3 \
    --bf16 False \
    --fp16 True \
    --seed 42 \
    --prediction_loss_only True \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --use_peft False \
    --lora_r 32 \
    --lora_a 64 \
    --lora_dropout 0.1 \
    --lora_task_type "CAUSAL_LM" \
    --torch_dtype "float16" \
    --add_special_tokens False \
    --append_concat_token False \
    --gradient_checkpointing True \
    --lr_scheduler_type "cosine" \
    --packing False \
    # --use_reentrant True \
    # --packing
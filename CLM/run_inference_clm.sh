CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 29500 \
    lm_inference.py \
    --task_name question_generation \
    --max_new_tokens 128 \
    --checkpoint "./question_generation/FT/gemma-2-9b/lora/fp16/e7/lr5e-5-hl-eos-r32" \
    --num_proc 4 \
    --base_model_name "google/gemma-2-9b-it" \
    --model_type gemma \
    --batch_size 4 \
    --num_beams 4 \
    --use_peft
    # --min_new_tokens 110 \

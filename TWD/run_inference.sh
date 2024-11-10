CUDA_VISIBLE_DEVICES=2,3  accelerate launch --main_process_port 29501 \
    inference_multi.py \
    --task question_generation \
    --max_source_length 512 \
    --max_target_length 128 \
    --checkpoint "./question_generation/FT-lora/google-flan-t5-large/lr1e-4-r64" \
    --num_proc 4 \
    --base_model_name "google/flan-t5-large" \
    --batch_size 32 \
    --num_beams 4 \
    --use_lora
    # --min_new_tokens 110 \

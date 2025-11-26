CUDA_VISIBLE_DEVICES=0  accelerate launch --main_process_port 29500 \
    lm_inference_no_ft.py \
    --task_name question_generation \
    --max_new_tokens 128 \
    --num_proc 4 \
    --base_model_name "google/gemma-2-9b-it" \
    --output_dir "./question_generation/NO-FT/google-gemma-2-9b-it/zeroshot-base-b8-beam4-no-hl-passage-answer" \
    --model_type gemma \
    --batch_size 4 \
    --num_beams 4 \
    # --use_peft
    # --min_new_tokens 110 \
    # meta-llama/Llama-3.1-8B-Instruct
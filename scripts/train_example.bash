CUDA_VISIBLE_DEVICES=0,1 python \
    bitdelta/train.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --finetuned_model lmsys/vicuna-7b-v1.5 \
    --save_dir $MODEL_SAVE_DIR \
    --batch_size 4 \
    --num_steps 200 \
 #   --save_full_model True
CUDA_VISIBLE_DEVICES=0 python \
    bitdelta/eval_ppl.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --dataset_name wikitext \
    --subset wikitext-2-raw-v1 \
    --save_dir $PPL_SAVE_DIR \
    --num_eval_samples 100 \
    --model_diff $MODEL_SAVE_DIR/diff.pt \
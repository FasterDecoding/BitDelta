python \
    bitdelta/train.py \
    --base_model 'meta-llama/Llama-2-70b-hf' \
    --finetuned_model 'meta-llama/Llama-2-70b-chat-hf' \
    --save_dir $SAVE_DIR \
    --batch_size 1 \
    --num_steps 800 \
    --base_model_device 'cpu' \
    --finetuned_model_device '0,1' \
    --finetuned_model_memory_map "{\"0\": \"150GiB\", \"1\": \"150GiB\"}" \
    --finetuned_compressed_model_device '2,3,4,5,6,7' \
    --finetuned_compressed_model_memory_map "{\"2\": \"50GiB\", \"3\": \"50GiB\", \"4\": \"50GiB\", \"5\": \"50GiB\", \"6\": \"50GiB\", \"7\": \"50GiB\"}" \
#   --save_full_model True
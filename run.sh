MODEL_SAVE_DIR=/home/pingbowen/workspace/delta-compression/save/test

mkdir -p $MODEL_SAVE_DIR

values=(0.05 0.2 0.4 0.5 0.75)

pretrained_model=(/data/public/opensource_models/codellama/codellama-7b-python-hf/ /data/public/opensource_models/meta-llama/Llama-2-7b-hf/)
finetuned_model=(/data/groups/QY_LLM_Other/OSS_Code_LLM/Magicoder-S-CL-7B/ /data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf/)
svd_dict=(/home/pingbowen/workspace/delta-compression/saved_model/magicoder_svd.pt /home/pingbowen/workspace/delta-compression/saved_model/llama_chat_svd.pt)
for (( i=0; i<2; i++ )); do

gpu0=$((2 * i))
gpu1=$((2 * i + 1))

CUDA_VISIBLE_DEVICES="$gpu0,$gpu1" python \
    bitdelta/train2.py \
    --base_model ${pretrained_model[$i]} \
    --finetuned_model ${finetuned_model[$i]} \
    --save_dir $MODEL_SAVE_DIR \
    --batch_size 4 \
    --num_steps 200 \
    --save_full_model True \
    --attn_outlier 0.2 \
    --mlp_outlier 0.1 \
    --svd_dict ${svd_dict[$i]} \
    --choice svd &
    # &> test.log
done
wait
    #  /data/public/opensource_models/codellama/codellama-7b-python-hf/ /data/groups/QY_LLM_Other/OSS_Code_LLM/Magicoder-S-CL-7B/
    #  /home/pingbowen/models/vicuna-13b-v1.5 , /home/pingbowen/models/Llava-v1.5
    # /data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/
    # /data/public/opensource_models/meta-llama/Llama-2-7b-hf/ /data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf/

MODEL_SAVE_DIR=/home/pingbowen/workspace/delta-compression/save/mistral-v0.2_bitdelta

mkdir -p $MODEL_SAVE_DIR

values=(0.05 0.2 0.4 0.5 0.75)

pretrained_model=(/data/public/opensource_models/meta-llama/Llama-2-7b-hf/ )
finetuned_model=(/data/public/wangshuo/exp/ft-en-magicoder-llama-2-7b/ckpts/checkpoints/epoch_2_hf
)
svd_dict=(/data/groups/QY_LLM_Other/pingbowen/models/codelora/codelora_svd.pt / )
save_dir=(/home/pingbowen/workspace/delta-compression/save/test /data/groups/QY_LLM_Other/pingbowen/models/codelora/codelora_bitdelta/)

for (( i=0; i<2; i++ )); do

# choice="svd"
if [ $i -eq 0 ]; then
    choice="svd"
else
    choice="bit"
fi

gpu0=$((2 * i))
gpu1=$((2 * i + 1))
# "$gpu0,$gpu1"
CUDA_VISIBLE_DEVICES=$((i + 1)) python \
    bitdelta/train2.py \
    --base_model ${pretrained_model[0]} \
    --finetuned_model ${finetuned_model[0]}  \
    --save_dir ${save_dir[$i]} \
    --batch_size 4 \
    --num_steps 200 \
    --save_full_model True \
    --attn_outlier 0.2 \
    --mlp_outlier 0.1 \
    --svd_dict ${svd_dict[$i]} \
    --dim 1024 \
    --scale_factor 1.46 \
    --choice $choice &
    # &> test.log # ${save_dir[$i]}
done
wait
    #  /data/public/opensource_models/codellama/codellama-7b-python-hf/ /data/groups/QY_LLM_Other/OSS_Code_LLM/Magicoder-S-CL-7B/
    #  /home/pingbowen/models/vicuna-13b-v1.5 , /home/pingbowen/models/Llava-v1.5
    # /data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/
    # /data/public/opensource_models/meta-llama/Llama-2-7b-hf/ /data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf/

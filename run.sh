MODEL_SAVE_DIR=./../save/test

mkdir -p $MODEL_SAVE_DIR

values=(0.05 0.2 0.4 0.5 0.75)

# for value in ${values[@]}
# do
CUDA_VISIBLE_DEVICES=5,6 python \
    bitdelta/train2.py \
    --base_model /data/public/opensource_models/meta-llama/Llama-2-7b-hf/ \
    --finetuned_model  /data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/ \
    --save_dir $MODEL_SAVE_DIR \
    --batch_size 4 \
    --num_steps 200 \
    --save_full_model True \
    --attn_outlier 0.2 \
    --mlp_outlier 0.1 \
    --choice bit
    # &> test.log
# done

    #  /data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf/
    #  /home/pingbowen/models/vicuna-13b-v1.5 , /home/pingbowen/models/Llava-v1.5
    # /data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/

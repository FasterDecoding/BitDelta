CUDA_VISIBLE_DEVICES=2,3 python tailor.py \
  --pretrained_model_name /data/public/opensource_models/meta-llama/Llama-2-7b-hf/ \
  --finetuned_model_name /data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf/\
  --dim 128 \
  --scale_factor 1.45 \
  --save_dir /home/pingbowen/save/Llama-2-7b-chat_svd
  
  
# &
# /data/public/opensource_models/codellama/codellama-7b-python-hf/
# /data/groups/QY_LLM_Other/OSS_Code_LLM/Magicoder-S-CL-7B/
# python3 tailor.py \
#   --finetuned_model_name /data/public/wangshuo/exp/ft-en-metameth-llama-2-7b/ckpts/checkpoints/epoch_2_hf \
#   --save_dir /home/pingbowen/workspace/delta-compression/BitDelta/tailor_model/math_lora_7b \
pretrained_model=(/data/public/opensource_models/meta-llama/Llama-2-7b-hf/ /data/public/opensource_models/meta-llama/Llama-2-7b-hf/ /data/public/opensource_models/codellama/codellama-7b-python-hf/  /home/pingbowen/models/vicuna-7b-v1.5)
finetuned_model=(/data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/ /data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf/ /data/groups/QY_LLM_Other/OSS_Code_LLM/Magicoder-S-CL-7B/  /home/pingbowen/models/llava-v1.5-7b)
finetuned_compressed_model=(/home/pingbowen/workspace/delta-compression/saved_model/WizardMath-7B-V1.0_bitdelta/ /data/groups/QY_LLM_Other/pingbowen/models/wizardmath/WizardMath_svd/ /data/groups/QY_LLM_Other/pingbowen/models/wizardmath/delta_1024_mix_32_8_3_2_full/)
param_types=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)
model_types=(svd bitdelta mix)

for (( j=0; j<${#param_types[@]}; j++ )); do
  CUDA_VISIBLE_DEVICES=1 python3 Plot.py --param_type ${param_types[$j]}
done
# for i in {0..2} 
# do
#   for (( j=0; j<${#param_types[@]}; j++ )); do
#     CUDA_VISIBLE_DEVICES=1,7 python tailor.py \
#       --pretrained_model_name ${pretrained_model[0]} \
#       --finetuned_model_name ${finetuned_model[0]} \
#       --finetuned_compressed_model ${finetuned_compressed_model[$i]} \
#       --dim 128 \
#       --scale_factor 1.45 \
#       --param_type ${param_types[$j]} \
#       --model_type ${model_types[$i]} \
#       --save_dir ./statistic/
#   done
# done
  
  
# &
# /data/public/opensource_models/codellama/codellama-7b-python-hf/
# /data/groups/QY_LLM_Other/OSS_Code_LLM/Magicoder-S-CL-7B/
# python3 tailor.py \
#   --finetuned_model_name /data/public/wangshuo/exp/ft-en-metameth-llama-2-7b/ckpts/checkpoints/epoch_2_hf \
#   --save_dir /home/pingbowen/workspace/delta-compression/BitDelta/tailor_model/math_lora_7b \
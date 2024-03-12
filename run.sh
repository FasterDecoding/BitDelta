MODEL_SAVE_DIR=save/uncalibrated_model_0

mkdir -p $MODEL_SAVE_DIR

CUDA_VISIBLE_DEVICES=6,7 python \
    bitdelta/train2.py \
    --base_model /data/public/opensource_models/meta-llama/Llama-2-7b-hf/ \
    --finetuned_model /data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/ \
    --save_dir $MODEL_SAVE_DIR \
    --batch_size 4 \
    --num_steps 200 \
    --save_full_model True 

    # --layers "layers.5."\
    # /data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/

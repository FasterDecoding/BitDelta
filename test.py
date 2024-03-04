import argparse
import transformers
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights
import torch.nn as nn
import os
from llava.model.language_model.llava_llama import LlavaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.model import *

def get_tokenizer(tokenizer_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=False, 
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    return tokenizer

@torch.no_grad()
def load_diff(model, diff_dir):
    device = model.device
    diff_dict = torch.load(diff_dir)

    for name, module in model.named_modules():
        if name + ".mask" in diff_dict:
            coeff = diff_dict[name + ".coeff"].to(device)
            mask = diff_dict[name + ".mask"].to(device)

            setattr(module, "mask", mask)
            setattr(module, "coeff", coeff)
            # module.weight.add_((mask * coeff).to(module.weight.dtype))
        elif name + ".weight" in diff_dict:
            module.weight = nn.Parameter(diff_dict[name + ".weight"].to(device).to(module.weight.dtype))

        elif name + '.A' in diff_dict:
            A = diff_dict[name + '.A'].to(device)
            B = diff_dict[name + '.B'].to(device)

            mask = (A @ B).T
            module.weight.add_(mask.to(module.weight.dtype))

    model.config.vocab_size = model.lm_head.weight.size(0)


def get_model(model_name, device, memory_map=None):
    # multi-gpu
    if device == "auto" or isinstance(device, list):
        
        # if gpus are specified, distributes according to the memory map
        if isinstance(device, list):
            assert memory_map is not None, "memory_map must be specified when using multiple gpus"
            config = AutoConfig.from_pretrained(model_name)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)

            device_map = infer_auto_device_map(model, memory_map, no_split_module_classes=["LlamaDecoderLayer"])

        else:
            # use all available gpus
            device_map = "auto"

        return transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    else: # single-gpu or cpu
        return transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )


def save_full_model(base_model_name, finetuned_model_name, diff_dir, save_dir, device):
    base_model = get_model(base_model_name, device)
    tokenizer = get_tokenizer(finetuned_model_name)
    load_diff(base_model, diff_dir)

    base_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    del base_model

model_path = "/home/pingbowen/models/Llava-v1-vicuna/Llava-v1/"

lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
print('Loading LLaVA from base model...')
model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
if model.lm_head.weight.shape[0] != token_num:
    model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))



# base_model = get_model("/home/pingbowen/models/Llava-v1-vicuna/Llava-v1/", "cuda")
# params = base_model.state_dict()

# print(params.keys())

# get_tokenizer("/data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/")
# save_full_model("/data/public/opensource_models/meta-llama/Llama-2-7b-hf/", "/data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/", os.path.join("/home/pingbowen/workspace/delta-compression/BitDelta/save", "diff_untrained.pt"), os.path.join("/home/pingbowen/workspace/delta-compression/BitDelta/save", "uncalibrated_model"), device="cuda")
import argparse
import jsonlines
import sys
import shutil
import logging
import os
import time
from tqdm import tqdm
import glob
import json
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
# from vllm import LLM, SamplingParams
import re
import random
import numpy as np

pretrained_model_name = "/data/public/opensource_models/meta-llama/Llama-2-7b-hf"

finetuned_model_name = "/data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf" # /data/public/wangshuo/exp/ft-en-magicoder-llama-2-7b/ckpts/checkpoints/epoch_2_hf

pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name,
                                        device_map="cpu")
pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)
finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=finetuned_model_name,
                                     device_map="cpu")
finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=finetuned_model_name)

save_dir = "/home/pingbowen/workspace/delta-compression/BitDelta/save/uncalibrated_model"

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(seed=0)
# scale_factor = finetuned_model.config.intermediate_size / finetuned_model.config.hidden_size


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge



task_vector_param_dict = {}
pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
# param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=[])
# with torch.no_grad():
#     for param_name in finetuned_param_dict.keys():
#         task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]
#         print(f"name {param_name} data {task_vector_param_dict[param_name]} ")


# import pdb
# pdb.set_trace()

def decomposition(masked_input_tensor,dim):

    U , S , V = torch.svd(masked_input_tensor)
    U , S , V = U[:, :dim],S[:dim],V[:, :dim]
    # return torch.mm(U, torch.diag(S)), V.t()
    # return U, torch.mm(torch.diag(S), V.t())   #return lora_B, lora_A
    return torch.mm(torch.mm(U, torch.diag(S)), V.t())

# dim = 256
dim = 128
# dim = 16
print("----------------------dim: ",dim)
print("----------------------dim: ",dim)
print("----------------------dim: ",dim)
print("----------------------dim: ",dim)
print("----------------------dim: ",dim)
print("----------------------dim: ",dim)

peft_dict = {}
malign_dict = {}
other_dict = {}

# finetuned_param_dict
# for param_name, param_value in tqdm(pretrained_param_dict.items()):
#     if "self_attn" in param_name or "mlp" in param_name:
#         pass
#     else:
#         other_dict[param_name] = param_value.contiguous()

diff = dict()

for param_name, param_value in tqdm(finetuned_param_dict.items()):
    if "self_attn" in param_name or "mlp" in param_name:
        delta = param_value - pretrained_param_dict[param_name]
        if "mlp" in param_name:
            dim = int(dim * 1.45)
        delta = decomposition(delta,dim=dim)
        diff[param_name] = (pretrained_param_dict[param_name] + delta).contiguous()
    else:
        diff[param_name] = param_value.contiguous()
        # lora_A = lora_A * (dim/16)  ###补偿scaling, 以后的alpha可以统一为16
        # peft_key = "base_model.model." + param_name.split(".weight")[0]
        # print(peft_key+".lora_A.weight")
        # peft_dict[peft_key+".lora_A.weight"] = lora_A.contiguous()
        # peft_dict[peft_key+".lora_B.weight"] = lora_B.contiguous()

for n,p in pretrained_model.named_parameters():
    p.data.copy_(diff[n])
    
pretrained_model.save_pretrained(save_dir)
finetuned_tokenizer.save_pretrained(save_dir)

# other_dict = {k: v.to(torch.float16) for k, v in other_dict.items()}

# other_para_path = "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/code/other_param"
# torch.save(other_dict, os.path.join(other_para_path, "other.pt"))
# torch.save(other_dict, os.path.join(other_para_path, "pretrain_other.pt"))


# peft_dict = {k: v.to(torch.float16) for k, v in peft_dict.items()}

# layernum = 40
# for lnum in range(layernum):
#     peft_pfx = f"base_model.model.model.layers.{lnum}"
#     delta_pfx = f"encoder.layers.{lnum}" 
#     malign_dict[f"{delta_pfx}.self_att.self_attention.project_q_lora.lora_A.weight"] = peft_dict[f"{peft_pfx}.self_attn.q_proj.lora_A.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.self_att.self_attention.project_q_lora.lora_B.weight"] = peft_dict[f"{peft_pfx}.self_attn.q_proj.lora_B.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.self_att.self_attention.project_k_lora.lora_A.weight"] = peft_dict[f"{peft_pfx}.self_attn.k_proj.lora_A.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.self_att.self_attention.project_k_lora.lora_B.weight"] = peft_dict[f"{peft_pfx}.self_attn.k_proj.lora_B.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.self_att.self_attention.project_v_lora.lora_A.weight"] = peft_dict[f"{peft_pfx}.self_attn.v_proj.lora_A.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.self_att.self_attention.project_v_lora.lora_B.weight"] = peft_dict[f"{peft_pfx}.self_attn.v_proj.lora_B.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.self_att.self_attention.attention_out_lora.lora_A.weight"] = peft_dict[f"{peft_pfx}.self_attn.o_proj.lora_A.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.self_att.self_attention.attention_out_lora.lora_B.weight"] = peft_dict[f"{peft_pfx}.self_attn.o_proj.lora_B.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.ffn.ffn.w_in.w_0_lora.lora_A.weight"] = peft_dict[f"{peft_pfx}.mlp.gate_proj.lora_A.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.ffn.ffn.w_in.w_0_lora.lora_B.weight"] = peft_dict[f"{peft_pfx}.mlp.gate_proj.lora_B.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.ffn.ffn.w_in.w_1_lora.lora_A.weight"] = peft_dict[f"{peft_pfx}.mlp.up_proj.lora_A.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.ffn.ffn.w_in.w_1_lora.lora_B.weight"] = peft_dict[f"{peft_pfx}.mlp.up_proj.lora_B.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.ffn.ffn.w_out_lora.lora_A.weight"] = peft_dict[f"{peft_pfx}.mlp.down_proj.lora_A.weight"].contiguous()
#     malign_dict[f"{delta_pfx}.ffn.ffn.w_out_lora.lora_B.weight"] = peft_dict[f"{peft_pfx}.mlp.down_proj.lora_B.weight"].contiguous()





# malign_dict = {k: v.to(torch.float16) for k, v in malign_dict.items()}

# import pdb
# pdb.set_trace()

output_peft_path = "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim256_2/code"
output_malign_path = "/home/wanghanqing/projects/exp/mAlign_exp/mAlign_LoRAs/trim_lora/dim256_2/code"

# torch.save(peft_dict, os.path.join(output_peft_path, "adapter_model.bin"))
# torch.save(malign_dict, os.path.join(output_malign_path, "lora.pt"))


print("--end--")


# for param_name, param_value in finetuned_model.named_parameters():
#     if param_name in masked_param_dict:
#         param_value.data.copy_(masked_param_dict[param_name])

# logger.info(f"saving model at {save_model_path}...")
# os.makedirs(save_model_path, exist_ok=True)
# finetuned_model.save_pretrained(save_directory=save_model_path)
# finetuned_tokenizer.save_pretrained(save_directory=save_model_path)
# logger.info(f"model is saved")
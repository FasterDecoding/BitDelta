import argparse
import transformers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights
import torch.nn as nn
import torch.nn.functional as F
# from llava.model.language_model.llava_llama import LlavaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
# from llava.model import *

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


device = "cuda" if torch.cuda.is_available() else "cpu"

a = torch.rand(4096) / 1000
b = torch.rand(4096) / 1000

# a , b = a.to(torch.bfloat16) , b.to(torch.bfloat16)

dot_fp , dot_pp = torch.dot(a, b) , torch.dot(b, b)

x = dot_fp / dot_pp

cosine_sim = F.cosine_similarity(a,b,dim=0)

cosine_sim2 = F.cosine_similarity(b,a - x * b,dim=0)

def filter_top_and_bottom_percent(tensor, percent=0.5):
    # 计算保留的元素数量
    num_elements = tensor.numel()
    num_to_keep = int(num_elements * percent / 100)

    # 展平张量并获取最大和最小的元素的索引
    flat_tensor = tensor.flatten()
    _, top_indices = torch.topk(flat_tensor, num_to_keep, largest=True)
    _, bottom_indices = torch.topk(flat_tensor, num_to_keep, largest=False)

    # 创建一个全零张量
    result = torch.zeros_like(tensor)

    # 仅在指定位置放置最大和最小的元素
    result = result.flatten()
    result[top_indices] = flat_tensor[top_indices]
    result[bottom_indices] = flat_tensor[bottom_indices]
    result = result.reshape(tensor.shape)

    return result

def copy_nonzero_values(A, B):
    # 复制B中非零值到A的对应位置
    mask = B != 0
    A[mask] = B[mask]
    return A

def load_svd(model):
    param_dict = torch.load(args.svd_dict)
    # import pdb; pdb.set_trace() 
    with torch.no_grad():
        for k,v in param_dict.items():
            if "base" in k:
                dim = args.dim
                
                if "mlp" in k:
                    dim = int(dim * args.scale_factor)
                
                k = k.replace(".base", "")
                
                U = param_dict[k + ".U"][:, :dim]
                S = param_dict[k + ".S"][:dim]
                V = param_dict[k + ".V"][:, :dim]
                # import pdb; pdb.set_trace()
                model.get_submodule(k).weight.copy_(v + U @ torch.diag(S) @ V.t())

parser = argparse.ArgumentParser(description="BitDelta")
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--scale_factor", type=float, default=1.45)
parser.add_argument("--svd_dict", type=str, default="")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("/data/public/wangshuo/exp/ft-en-metameth-llama-2-7b/ckpts/checkpoints/epoch_2_hf")
model = AutoModelForCausalLM.from_pretrained("/data/public/wangshuo/exp/ft-en-metameth-llama-2-7b/ckpts/checkpoints/epoch_2_hf", torch_dtype=torch.bfloat16) # low_cpu_mem_usage=True

load_svd(model)

tokenizer.save_pretrained(f"/data/groups/QY_LLM_Other/pingbowen/models/mathlora/math_svd/")
model.save_pretrained(f"/data/groups/QY_LLM_Other/pingbowen/models/mathlora/math_svd/")

# get_tokenizer("/data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/")
# save_full_model("/data/public/opensource_models/meta-llama/Llama-2-7b-hf/", "/data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/", os.path.join("/home/pingbowen/workspace/delta-compression/BitDelta/save", "diff_untrained.pt"), os.path.join("/home/pingbowen/workspace/delta-compression/BitDelta/save", "uncalibrated_model"), device="cuda")
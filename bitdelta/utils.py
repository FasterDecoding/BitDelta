import argparse
import transformers
import torch
from transformers import AutoConfig, AutoModelForCausalLM,AutoTokenizer
from accelerate import infer_auto_device_map, init_empty_weights
import os
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def load_llava(path,device):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    
    image_processor = None

    if 'llava' in path.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device)
        if device != 'auto':
            vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="BitDelta")
    
    # models
    parser.add_argument(
        "--finetuned_model", type=str, default="lmsys/vicuna-7b-v1.5-16k"
    )
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf")

    # train params
    parser.add_argument("--dataset_name", type=str, default="c4")
    parser.add_argument("--subset", type=str, default="en")
    parser.add_argument("--data_dir", type=str, default="en")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--layers", nargs='+', default=None)
    parser.add_argument("--save_num", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--attn_outlier", type=float,default=1e-4)
    parser.add_argument("--mlp_outlier", type=float,default=1e-4)
    parser.add_argument("--choice", type=str,choices=['mix','bit','rank'],default=None)

    # device management
    parser.add_argument("--base_model_device", type=str, default="0")
    parser.add_argument("--finetuned_model_device", type=str, default="0")
    parser.add_argument("--finetuned_compressed_model_device", type=str, default="1")
    parser.add_argument("--save_full_model", type=bool, default=False)

    # multi-gpu support (assumes fp32, should 2x since we use bf16)
    # Example: --finetuned_model_memory_map "{\"0\": \"150GiB\", \"1\": \"150GiB\"}"
    parser.add_argument("--base_model_memory_map", type=str, default=None)
    parser.add_argument("--finetuned_model_memory_map", type=str, default=None)
    parser.add_argument("--finetuned_compressed_model_memory_map", type=str, default=None)

    # ppl eval params
    parser.add_argument("--num_eval_samples", type=int, default=10)
    parser.add_argument("--context_size", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--model_diff", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    parser.add_argument("--debug", type=bool, default=False)

    args = parser.parse_args()

    args.base_model_device = parse_device(args.base_model_device)
    args.finetuned_model_device = parse_device(args.finetuned_model_device)
    args.finetuned_compressed_model_device = parse_device(
        args.finetuned_compressed_model_device
    )
    
    args.base_model_memory_map = parse_dict(args.base_model_memory_map)
    args.finetuned_model_memory_map = parse_dict(args.finetuned_model_memory_map)
    args.finetuned_compressed_model_memory_map = parse_dict(
        args.finetuned_compressed_model_memory_map
    )

    return args

def parse_dict(d: str):
    if d is None:
        return d
    import json
    # print(d)
    res = json.loads(d)
    return {int(k): v for k, v in res.items()}
def parse_device(device: str):
    if ',' in device:
        return [int(d) for d in device.split(',')]
    elif device in ["auto", "cpu"]:
        return device
    return f"cuda:{device}"

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
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)


def get_tokenizer(tokenizer_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=False, trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    return tokenizer

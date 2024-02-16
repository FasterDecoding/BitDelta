import argparse
import transformers
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights

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
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--save_dir", type=str, required=True)


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

import os

import torch

import torch.nn.functional as F
from bitdelta.diff2 import compress_diff, save_diff, save_full_model
from bitdelta.misc import find_corr_stddev

from bitdelta.utils import get_model, parse_args, get_tokenizer
from tqdm import tqdm
from bitdelta.data import get_dataset, get_dataloader

import json

args = parse_args()

# create save_dir if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)

tokenizer = get_tokenizer(args.finetuned_model)

with torch.no_grad():
    base_model = get_model(args.base_model, args.base_model_device, args.base_model_memory_map).to(torch.float32)
    finetuned_model = get_model(args.finetuned_model, args.finetuned_model_device, args.finetuned_model_memory_map).to(torch.float32)

finetuned_compressed_model = get_model(args.finetuned_model, args.finetuned_compressed_model_device, args.finetuned_compressed_model_memory_map)

print(f"compressing diff...")
compress_diff(base_model, finetuned_model, finetuned_compressed_model,args.save_dir)

tokenizer.save_pretrained(args.save_dir)

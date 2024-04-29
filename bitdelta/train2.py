import os

import torch

import torch.nn.functional as F
from bitdelta.diff2 import compress_diff, save_diff, save_full_model
from bitdelta.misc import find_corr_stddev

from bitdelta.utils import get_model, parse_args, get_tokenizer,load_llava
from tqdm import tqdm
from bitdelta.data import get_dataset, get_dataloader

import json

args = parse_args()

# create save_dir if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)

tokenizer = get_tokenizer(args.finetuned_model)

with torch.no_grad():
    base_model = get_model(args.base_model, args.base_model_device, args.base_model_memory_map)
    if "llava" in args.finetuned_model.lower():
        finetuned_model = load_llava(args.finetuned_model,device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        finetuned_model = get_model(args.finetuned_model, args.finetuned_model_device, args.finetuned_model_memory_map)


print(f"compressing diff...")
compress_diff(base_model, finetuned_model, args.save_dir,args)

tokenizer.save_pretrained(args.save_dir)
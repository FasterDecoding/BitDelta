import os

import torch

import torch.nn.functional as F
from bitdelta.diff import compress_diff, save_diff, save_full_model
from bitdelta.misc import find_corr_stddev

from bitdelta.utils import get_model, parse_args, get_tokenizer
from tqdm import tqdm
from bitdelta.data import get_dataset, get_dataloader

import json

args = parse_args()

# create save_dir if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)

tokenizer = get_tokenizer(args.base_model)

with torch.no_grad():
    base_model = get_model(args.base_model, args.base_model_device, args.base_model_memory_map)
    finetuned_model = get_model(args.finetuned_model, args.finetuned_model_device, args.finetuned_model_memory_map)

finetuned_compressed_model = get_model(args.finetuned_model, args.finetuned_compressed_model_device, args.finetuned_compressed_model_memory_map)

print(f"compressing diff...")
compress_diff(base_model, finetuned_model, finetuned_compressed_model)

# save untrained delta
save_diff(finetuned_compressed_model, os.path.join(args.save_dir, "diff_untrained.pt"))


if args.save_full_model:
    print("saving uncalibrated model")
    save_full_model(args.base_model, args.finetuned_model, os.path.join(args.save_dir, "diff_untrained.pt"), os.path.join(args.save_dir, "uncalibrated_model"), device="cpu")

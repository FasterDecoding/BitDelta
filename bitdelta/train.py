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

def original_diff(base_model, finetuned_model):
    origin_diff = {}
    for k, v in finetuned_model.named_parameters():
        if "mlp" in k or "self_attn" in k:
            origin_diff[k] = v.detach().cpu() - base_model.get_submodule(k.replace('.weight',"")).weight.detach().cpu()
    return origin_diff

# get corr/stddev stats
if args.debug:
    print(f"finding corr/stddev stats...")
    corrs, stddevs = find_corr_stddev(base_model, finetuned_model)
    corr = sum(corrs) / len(corrs)
    stddev = sum(stddevs) / len(stddevs)
    # save in args.save_dir as csv
    with open(os.path.join(args.save_dir, "corr_stddev.csv"), "w") as f:
        f.write(f"corr,stddev\n{corr},{stddev}")


finetuned_compressed_model = get_model(args.finetuned_model, args.finetuned_compressed_model_device, args.finetuned_compressed_model_memory_map)

print(f"compressing diff...")
compress_diff(base_model, finetuned_model, finetuned_compressed_model,layers=args.layers)

tokenizer.save_pretrained("/home/pingbowen/workspace/delta-compression/BitDelta/save/test")


'''
train_num_samples = args.batch_size * args.num_steps
train_dataset = get_dataset(
    args.dataset_name,
    args.subset,
    "train",
    size=train_num_samples,
)
train_dataloader = get_dataloader(
    train_dataset,
    tokenizer,
    args.batch_size,
    num_workers=4,
    max_length=args.max_length,
)

# save untrained delta
save_diff(finetuned_compressed_model, os.path.join(args.save_dir, "diff_untrained.pt"),layers=args.layers)

if args.train:
    optimizer = torch.optim.AdamW(finetuned_compressed_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_steps)

    bar = tqdm(train_dataloader)

    train_loss_list = []

    # Train loop
    for step, batch in enumerate(bar):
        batch1 = {k: v.to(finetuned_model.device) for k, v in batch.items()}
        with torch.inference_mode():
            finetuned_outputs = finetuned_model(**batch1)

        batch2 = {k: v.to(finetuned_compressed_model.device) for k, v in batch.items()}
        finetuned_compressed_outputs = finetuned_compressed_model(**batch2)

        loss = F.mse_loss(
            finetuned_outputs.logits.clone().to(finetuned_compressed_outputs.logits.device),
            finetuned_compressed_outputs.logits,
        )

        train_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        bar.set_description(f"train loss: {loss.item()}")


# save loss list
if args.debug:
    with open(os.path.join(args.save_dir, f"train_loss_{args.num_groups}.json"), "w") as f:
        json.dump(train_loss_list, f)

ori_diff = original_diff(base_model, finetuned_model)

# # save trained delta
save_diff(finetuned_compressed_model, os.path.join(args.save_dir, "diff.pt"),layers=args.layers)

del base_model, finetuned_model, finetuned_compressed_model
torch.cuda.empty_cache()

if args.save_full_model:
    print("saving uncalibrated model")
    save_full_model(args.base_model, args.finetuned_model, os.path.join(args.save_dir, "diff_untrained.pt"), os.path.join(args.save_dir, f"uncalibrated_model"), device="cpu",layers=args.layers,ori_diff=ori_diff)
    # print("saving calibrated model")
    # save_full_model(args.base_model, args.finetuned_model, os.path.join(args.save_dir, "diff.pt"), os.path.join(args.save_dir, "calibrated_model"), device="cpu")
'''
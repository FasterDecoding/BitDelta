from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm
import os
from bitdelta.utils import get_model, parse_args, get_tokenizer
from bitdelta.data import get_dataset
from bitdelta.diff import load_diff

device = "cuda" 

args = parse_args()

print("Loading dataset...")
dataset = get_dataset(
    args.dataset_name,
    args.subset,
    args.split,
    size=args.num_eval_samples,
)

text = ""
for sample in tqdm(dataset):
    text += sample["text"] + "\n\n"

print(text[:100])

tokenizer = get_tokenizer(args.base_model)

encodings = tokenizer(text, return_tensors="pt")

print(tokenizer.decode(encodings.input_ids[0][:100]))

max_length = args.context_size + args.window_size
stride = args.window_size
seq_len = encodings.input_ids.size(1)
# make seq_len a multiple of stride
seq_len = seq_len - (seq_len % stride)
print(f"seq_len: {seq_len}")

print(f"Loading model from {args.base_model}...")
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

if args.model_diff is not None:
    print("Loading diff...")
    load_diff(model, args.model_diff)

model_vocab_size = model.get_input_embeddings().weight.size(0)
tokenizer_vocab_size = len(tokenizer)

if model_vocab_size != tokenizer_vocab_size:
    if model_vocab_size != tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
    model.resize_token_embeddings(tokenizer_vocab_size)

model.eval()
nlls = []

pbar = tqdm(range(0, seq_len, stride))
for begin_loc in pbar:
    end_loc = begin_loc + max_length
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-stride] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    pbar.set_description(
        f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
    )

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc >= seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())

os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, "ppl.txt"), "w") as f:
    f.write(str(ppl.item()))

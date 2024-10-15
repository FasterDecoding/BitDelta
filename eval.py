import argparse
import transformers
import torch
from transformers import AutoConfig, AutoModelForCausalLM

def load_model(model_name):
    model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,)
    return model  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str)
    parser.add_argument('--finetuned_model', type=str)
    args = parser.parse_args()

    base_model = load_model(args.base_model)
    finetuned_model = load_model(args.finetuned_model)
    
    params = dict()
    
    for n,p in finetuned_model.named_parameters():
        if "mlp" in n or "self_attn" in n:
            delta = p - base_model.state_dict()[n]
            w = torch.sum(torch.abs(delta))
            params[n] = w.item()
    
    print(params)
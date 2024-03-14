import torch
import torch.nn as nn
import gc

from bitdelta.binary_gemm_kernel import pack, unpack, binary_bmm
from bitdelta.utils import get_model, get_tokenizer

class BinaryDiff(nn.Module):
    def __init__(self, weight):
        super().__init__()
        diff = weight
        quantile = diff.float().abs().mean()

        mask = torch.ones_like(diff)
        mask[diff < 0] = 0
        mask = pack(mask.bool().T)
     
        self.register_buffer("mask", mask)
        # self.register_buffer("base", base.T)
        self.register_parameter(
            "coeff",
            nn.Parameter(
                torch.tensor(
                    quantile,
                    dtype=torch.float32,
                    requires_grad=True,
                    device=weight.device,
                )
            ),
        )
        # del base, finetune, diff

    def forward(self, x):
        # print(x.shape, self.base.shape, self.coeff.shape, self.mask.shape)
        # [B, seq, in] @ [in, out] + [B, seq, in] @ [B, in/32, out]

        # TODO: This can be faster
        repeated_mask = self.mask.unsqueeze(0).repeat(x.size(0), 1, 1)
        return x @ self.base + self.coeff * binary_bmm(x, repeated_mask)

def Pass(layers=None,name=None):
    if layers is not None:
        for layer in layers:
            if layer in name:
                return True
    return False


def compress_diff(base_model, finetuned_model, finetuned_compressed_model,save_dir,layers=None):
    def compress_submodule(name, subname, module, submodule):
        target_device = submodule.weight.device
                    
        base_weight = base_model.get_submodule(f"{name}.{subname}").weight.detach().to(target_device)
        finetuned_weight = finetuned_model.get_submodule(f"{name}.{subname}").weight.detach().to(target_device)

        compressed = BinaryDiff(
            base=base_weight,
            finetune=finetuned_weight,
        ).to(target_device)

        del submodule, base_weight
        setattr(module, subname, None)
        gc.collect()
        torch.cuda.empty_cache()
        setattr(module, subname, compressed)

    # TODO: 根据thresh 选择压缩比例
    for name, module in finetuned_compressed_model.named_modules():
        if "self_attn" in name or "mlp" in name:
            for subname, submodule in module.named_children():
                if "proj" in subname:
                    base_weight = base_model.get_submodule(f"{name}.{subname}").weight.detach().to(submodule.weight.device)
                    finetuned_weight = finetuned_model.get_submodule(f"{name}.{subname}").weight.detach().to(submodule.weight.device)
                    dim , thresh = 1024,0.7
                    
                    if "mlp" in name:
                        dim , thresh = 2048 , 0.24
                    
                    U,S,V = decomposition(finetuned_weight - base_weight,dim=dim)
                    energy_total = torch.sum(S**2)
                    energy_top_percent = torch.sum(S[:50]**2)
                    ratio = energy_top_percent / energy_total
                    
                    compressed_U, compressed_V = BinaryDiff(weight=U[:,64:]).to(finetuned_weight.device), BinaryDiff(weight=V[:,64:]).to(finetuned_weight.device)
                    U_mask, U_coeff, V_mask, V_coeff = compressed_U.mask, compressed_U.coeff, compressed_V.mask, compressed_V.coeff
                    weight_U , weight_V = (unpack(U_mask)*2-1) * U_coeff, (unpack(V_mask)*2-1) * V_coeff
                    # import pdb; pdb.set_trace()
                    U[:,64:] , V[:,64:] = weight_U.T, weight_V.T   # 不确定是否有bug
                    delta = U @ torch.diag(S) @ V.t()
                    with torch.no_grad():
                        finetuned_model.get_submodule(f"{name}.{subname}").weight.copy_(base_weight + delta.to(base_weight.dtype)) 
    
    finetuned_model.save_pretrained(save_dir)

def save_diff(finetuned_compressed_model, save_dir,layers=None,ori_diff=None):
    diff_dict = {}

    for name, module in finetuned_compressed_model.named_modules():
        if isinstance(module, BinaryDiff):
            # diff_dict[name + ".mask"] = (module.mask == 1).bool().cpu()
            diff_dict[name + ".mask"] = module.mask.cpu()
            diff_dict[name + ".coeff"] = module.coeff.cpu()

    for name, param in finetuned_compressed_model.named_parameters():
        if param.requires_grad:
            diff_dict[name] = param.cpu()

    torch.save(diff_dict, save_dir)

@torch.no_grad()
def load_diff(model, diff_dir,ori_diff):
    device = model.device
    diff_dict = torch.load(diff_dir)
    # ori_diff = torch.load(ori_diff)

    for name, module in model.named_modules():
        if name + ".mask" in diff_dict:
            coeff = diff_dict[name + ".coeff"].to(device)
            mask = diff_dict[name + ".mask"].to(device)

            # setattr(module, "mask", mask)
            # setattr(module, "coeff", coeff)
            weight = (unpack(mask)*2-1) * coeff
            weight_fp16 = decomposition(ori_diff[name + ".weight"].to(torch.float32), dim=64).to(torch.bfloat16)
            # import pdb; pdb.set_trace()

            module.weight.add_(weight_fp16.to(module.weight.dtype) + weight.T.to(module.weight.dtype))
        elif name + ".weight" in diff_dict:
            module.weight = nn.Parameter(diff_dict[name + ".weight"].to(device).to(module.weight.dtype))
            
            # if "mlp" in name:
            #     import pdb; pdb.set_trace()

        elif name + '.A' in diff_dict:
            A = diff_dict[name + '.A'].to(device)
            B = diff_dict[name + '.B'].to(device)

            mask = (A @ B).T
            module.weight.add_(mask.to(module.weight.dtype))

    model.config.vocab_size = model.lm_head.weight.size(0)

def decomposition(masked_input_tensor,dim=None,st=None,ed=None,name=None):
    U , S , V = torch.svd(masked_input_tensor.to(torch.float32))
    
    if dim is not None:
        U , S , V = U[:, :dim],S[:dim] ,V[:, :dim]
    
    if st is not None and ed is not None:
        U , S , V = U[:, st:ed],S[st:ed] ,V[:, st:ed]
    
    return U, S, V

def save_full_model(base_model_name, finetuned_model_name, diff_dir, save_dir, device,layers=None,ori_diff=None):
    base_model = get_model(base_model_name, device)
    tokenizer = get_tokenizer(finetuned_model_name)
    
    finetuned_model = get_model(finetuned_model_name, device)
    # params = {}
        
    load_diff(base_model, diff_dir,ori_diff=ori_diff)
    
    base_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    del base_model

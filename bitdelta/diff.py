import torch
import torch.nn as nn
import gc

from bitdelta.binary_gemm_kernel import pack, unpack, binary_bmm
from bitdelta.utils import get_model, get_tokenizer

class BinaryDiff(nn.Module):
    def __init__(self, base, finetune):
        super().__init__()
        diff = finetune - base
        outlier = get_outlier(diff, percent=0.02)
        set_zero(diff, outlier)
        # import pdb; pdb.set_trace()
        quantile = diff.float().abs().mean()

        mask = torch.ones_like(diff)
        mask[diff < 0] = 0
        mask = pack(mask.bool().T)
     
        self.register_buffer("mask", mask)
        self.register_buffer("base", base.T)
        self.register_buffer("outlier", outlier)
        self.register_parameter(
            "coeff",
            nn.Parameter(
                torch.tensor(
                    quantile,
                    dtype=torch.float32,
                    requires_grad=True,
                    device=base.device,
                )
            ),
        )
        del base, finetune, diff

    def forward(self, x):
        # print(x.shape, self.base.shape, self.coeff.shape, self.mask.shape)
        # [B, seq, in] @ [in, out] + [B, seq, in] @ [B, in/32, out]

        # TODO: This can be faster
        repeated_mask = self.mask.unsqueeze(0).repeat(x.size(0), 1, 1)
        return x @ self.base + self.coeff * binary_bmm(x, repeated_mask)

def set_zero(A, B):
    # 复制B中非零值到A的对应位置
    mask = B != 0
    A[mask] = 0
    return A

def get_outlier(tensor, percent=0.5):
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

def compress_diff(base_model, finetuned_model, finetuned_compressed_model,layers=None):
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

    # TODO: this can be parallelized
    # flag = False
    with torch.no_grad():
        for name, module in finetuned_model.named_modules():
            if "self_attn" in name or "mlp" in name:
                for subname, submodule in module.named_children():
                    if "proj" in subname:
                        p , f = base_model.get_submodule(f"{name}.{subname}").weight.detach() , finetuned_model.get_submodule(f"{name}.{subname}").weight.detach()
                        
                        compressed = BinaryDiff(base=p, finetune=f)
                        mask, coeff , outlier = compressed.mask, compressed.coeff, compressed.outlier
                        weight = (unpack(mask)*2-1) * coeff
                        weight = weight.T.to(outlier.dtype)
                        
                        copy_nonzero_values(weight, outlier)
                        # import pdb; pdb.set_trace()
                        finetuned_model.get_submodule(f"{name}.{subname}").weight.copy_(p.to(p.dtype) + weight.to(p.dtype))
    
    finetuned_model.save_pretrained("/home/pingbowen/workspace/delta-compression/BitDelta/save/test")
                    
                    
                    

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

def decomposition(masked_input_tensor,dim=None,st=None,ed=None):
    U , S , V = torch.svd(masked_input_tensor.to(torch.float32))
    
    if dim is not None:
        U , S , V = U[:, :dim],S[:dim] ,V[:, :dim]
    
    if st is not None and ed is not None:
        U , S , V = U[:, st:ed],S[st:ed] ,V[:, st:ed]
    
    return torch.mm(torch.mm(U, torch.diag(S)), V.t())

def save_full_model(base_model_name, finetuned_model_name, diff_dir, save_dir, device,layers=None,ori_diff=None):
    base_model = get_model(base_model_name, device)
    tokenizer = get_tokenizer(finetuned_model_name)
    
    finetuned_model = get_model(finetuned_model_name, device)
    # params = {}
    
    # for k ,v in finetuned_model.named_parameters():
    #     if layers is not None:
    #         for layer in layers:
    #             if layer in k:
    #                 if "mlp" in k or "self_attn" in k:
    #                     delta =  v.detach().cpu() - base_model.get_submodule(k.replace('.weight',"")).weight.detach().cpu()
    #                     dim = 128
    #                     if "mlp" in k:  
    #                         dim = int(dim * 1.45)
    #                     # import pdb; pdb.set_trace()
    #                     params[k] = decomposition(delta.to(torch.float32), dim).to(torch.bfloat16)

    # dict(base_model.named_parameters())['model.layers.0.self_attn.o_proj.weight']
    
    # with torch.no_grad():
    #     for param in params:
    #         base_model.get_submodule(param.replace('.weight',"")).weight.add_(params[param].detach().to(device))
        
    load_diff(base_model, diff_dir,ori_diff=ori_diff)
    
    base_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    del base_model
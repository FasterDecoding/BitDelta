import torch
import torch.nn as nn
import gc
import torch.nn.functional as F
from bitdelta.binary_gemm_kernel import pack, unpack, binary_bmm
from bitdelta.utils import get_model, get_tokenizer

# 离群值抽出之后 原来位置设定成多少，如果设置成0会让分母增大
# U, V

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


def solve_orthogonal(p, f):
    # 计算x
    delta ,n , sacled_p = f - p, p.shape[-1],p

    # import pdb; pdb.set_trace()
    
    for i in range(n):
        p_i,f_i = p[:,i],f[:,i]
        dot_fp , dot_pd = torch.dot(f_i, p_i) , torch.dot(p_i, delta[:,i])
        
        if dot_fp == 0 or dot_pd == 0: # p_i或f_i是零向量，因为低秩, 边界p_i与delta_i直接正交
            continue
        
        dot_pp = torch.dot(p_i, p_i)
        x = dot_fp / dot_pp if dot_pp != 0 else None

        
        # 计算(f - xp)
        with torch.no_grad():
            delta[:,i].data.copy_(f_i - x * p_i) if x is not None else None
            sacled_p[:,i].data.copy_(sacled_p[:,i].data * x) if x is not None else None
        
    # import pdb; pdb.set_trace()
    
    return delta , sacled_p

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

def compress_diff(base_model, finetuned_model, save_dir,args):
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
    param_dict = dict()
    for name, module in finetuned_model.named_modules():
        if "vision" in name:
            continue
        
        if "self_attn" in name or "mlp" in name:
            for subname, submodule in module.named_children():
                
                with torch.no_grad():
                    if "proj" in subname:
                        p = base_model.get_submodule(f"{name}.{subname}").weight.detach().to(submodule.weight.device)
                        f = finetuned_model.get_submodule(f"{name}.{subname}").weight.detach().to(submodule.weight.device)
                        
                        delta , outlier_U, outlier_V = f - p , None, None
                        
                        if args.choice == "mix":
                            dim , fp16_col = 1024, 64
                            
                            if "self_attn" in name:
                                U,S,V,outlier_U,outlier_V = decomposition(delta,dim=dim,name=name,attn_outlier=args.attn_outlier) 
                            else:
                                dim , fp16_col = 1024 , 128
                                # delta , scaled_p = solve_orthogonal(p, f)
                                U,S,V,outlier_U,outlier_V = decomposition(delta,dim=dim,name=name,mlp_outlier=args.mlp_outlier)
                                                    
                            compressed_U, compressed_V = BinaryDiff(weight=U[:,fp16_col:]).to(f.device), BinaryDiff(weight=V[:,fp16_col:]).to(f.device)
                            U_mask, U_coeff, V_mask, V_coeff = compressed_U.mask, compressed_U.coeff, compressed_V.mask, compressed_V.coeff
                            weight_U , weight_V = (unpack(U_mask)*2-1) * U_coeff, (unpack(V_mask)*2-1) * V_coeff
                            U[:,fp16_col:] , V[:,fp16_col:] = weight_U.T, weight_V.T

                            
                            if outlier_U is not None and outlier_V is not None:
                                copy_nonzero_values(U[:,fp16_col:], outlier_U) , copy_nonzero_values(V[:,fp16_col:], outlier_V) 
                                # import pdb; pdb.set_trace() 
                            
                            delta = U @ torch.diag(S) @ V.t()
                        elif args.choice == "bit":
                            compressed = BinaryDiff(weight=delta).to(f.device)
                            mask , coeff = compressed.mask, compressed.coeff
                            delta = (unpack(mask)*2-1) * coeff
                            delta = delta.T
                        elif args.choice == "svd":
                            dim = 1024
                            
                            if "mlp" in name:
                                dim = int(1024 * 1.45)
        
                            U , S , V = decomposition((f - p).clone().detach(),dim=dim)
                            param_dict[f"{name}.{subname}" + ".base"] = p
                            param_dict[f"{name}.{subname}" + ".U"] = U.to(p.dtype)
                            param_dict[f"{name}.{subname}" + ".S"] = S.to(p.dtype)
                            param_dict[f"{name}.{subname}" + ".V"] = V.to(p.dtype)                            
                            # if "llava" in args.finetuned_model.lower():
                            #     U , S , V = decomposition((f - p).clone().detach(),dim=1024)
                            #     param_dict[f"{name}.{subname}" + ".base"] = p
                            #     param_dict[f"{name}.{subname}" + ".U"] = U.to(p.dtype)
                            #     param_dict[f"{name}.{subname}" + ".S"] = S.to(p.dtype)
                            #     param_dict[f"{name}.{subname}" + ".V"] = V.to(p.dtype)
                            
                        finetuned_model.get_submodule(f"{name}.{subname}").weight.copy_(p.to(p.dtype) + delta.to(p.dtype))
    
    # if "llava" in args.finetuned_model.lower():
    #     torch.save(param_dict, "/home/pingbowen/workspace/delta-compression/saved_model/llava_svd.pt")                     
    if args.choice == "svd":
        torch.save(param_dict, args.svd_dict)
    
    
    finetuned_model.to(torch.bfloat16)
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

def set_zero(A, B):
    # 复制B中非零值到A的对应位置
    mask = B != 0
    A[mask] = 0
    return A


def decomposition(masked_input_tensor,dim=None,name=None,attn_outlier=0.1,mlp_outlier=0.1):
    U , S , V = torch.svd(masked_input_tensor.to(torch.float32))
    
    outlier_U , outlier_V = None, None
    
    if dim is not None:
        U , S , V = U[:, :dim],S[:dim] ,V[:, :dim]
    
    # if "self_attn" in name:
    #     outlier_U = get_outlier(U[:,64:], percent=attn_outlier)
    #     outlier_V = get_outlier(V[:,64:], percent=attn_outlier)
        
    #     set_zero(U[:,64:], outlier_U)
    #     # import pdb; pdb.set_trace()
    #     set_zero(V[:,64:], outlier_V)
        
    # else:
    #     outlier_U = get_outlier(U[:,128:], percent=mlp_outlier)
    #     outlier_V = get_outlier(V[:,128:], percent=mlp_outlier)
        
    #     set_zero(U[:,128:], outlier_U)
    #     set_zero(V[:,128:], outlier_V)
    
    # max_val, min_val, mean_abs_val = round(torch.max(U).item(),4), round(torch.min(U).item(),4), round(torch.mean(torch.abs(U)).item(),4)
                            
    # print(f"max_val {max_val} pos_min {round(torch.min(outlier[outlier > 0]).item(),4)} mean_abs_val {mean_abs_val} ratio {round(torch.min(outlier[outlier > 0]).item() / mean_abs_val,4)}")
    # import pdb; pdb.set_trace()
    return U, S, V # , outlier_U, outlier_V

def save_full_model(base_model_name, finetuned_model_name, diff_dir, save_dir, device,layers=None,ori_diff=None):
    base_model = get_model(base_model_name, device)
    tokenizer = get_tokenizer(finetuned_model_name)
    
    finetuned_model = get_model(finetuned_model_name, device)
    # params = {}
        
    load_diff(base_model, diff_dir,ori_diff=ori_diff)
    
    base_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    del base_model


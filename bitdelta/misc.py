import torch
import torch.nn as nn
import gc
from bitdelta.diff import BinaryDiff

def find_corr_stddev(base_model, finetuned_model):
    corrs = []
    stddevs = [] # store std dev of base weight - finetuned weight
    for name, module in base_model.named_modules():
        if "mlp" in name or "self_attn" in name:
            for subname, submodule in module.named_children():
                if "proj" in subname:
                    base_weight = base_model.get_submodule(name + "." + subname).weight.view(-1).to("cpu")
                    finetuned_weight = finetuned_model.get_submodule(name + "." + subname).weight.view(-1).to("cpu")
                    # find correlation, assuming unravelled
                    base_stddev = torch.std(base_weight).float()
                    finetuned_stddev = torch.std(finetuned_weight).float()

                    covar = torch.dot(base_weight, finetuned_weight).float() / base_weight.shape[0]
                    corr = covar / (base_stddev * finetuned_stddev)

                    if corr.item() >= 1:
                        stddev = torch.tensor(0)
                    else:
                        stddev = torch.sqrt(base_stddev**2 + finetuned_stddev**2 - 2 * corr * base_stddev * finetuned_stddev)
                    corrs.append(corr.item())


                    stddevs.append(stddev.item())

    return corrs, stddevs


class MixtralBinaryDiff(nn.Module):
    def __init__(self, w1, mean_w1, w2, mean_w2, w3, mean_w3):
        super().__init__()
        self.w1 = BinaryDiff(mean_w1, w1)
        self.w2 = BinaryDiff(mean_w2, w2)
        self.w3 = BinaryDiff(mean_w3, w3)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

def compress_mixtral_moe_diff(model):
    for name, module in model.named_modules():
        if name.endswith("block_sparse_moe"):
            # print(name)
            experts = module.experts
            new_experts = nn.ModuleList()
            mean_w1 = torch.mean(torch.stack([expert.w1.weight for expert in experts]), dim=0).detach()
            mean_w2 = torch.mean(torch.stack([expert.w2.weight for expert in experts]), dim=0).detach()
            mean_w3 = torch.mean(torch.stack([expert.w3.weight for expert in experts]), dim=0).detach()
            for expert in experts:
                w1 = expert.w1.weight
                w2 = expert.w2.weight
                w3 = expert.w3.weight
                new_expert = MixtralBinaryDiff(w1, mean_w1, w2, mean_w2, w3, mean_w3)
                new_experts.append(new_expert)

            del experts, mean_w1, mean_w2, mean_w3
            setattr(module, "experts", None)
            gc.collect()
            torch.cuda.empty_cache()
            setattr(module, "experts", new_experts)


def dequantize_model(model, quantized_model, quant_type):
    from bitsandbytes.nn.modules import Linear8bitLt
    from auto_gptq.nn_modules.qlinear.qlinear_exllama import QuantLinear
    def dequantize_8bit(layer: Linear8bitLt):
        return ((layer.weight.CB * layer.weight.SCB.unsqueeze(dim=1)) / 127).half().detach()

    def dequantize_4bit(layer: QuantLinear, group_size=16):
        def unpack_4bit_to_32bit_signed(qweight, qzeros):
            # Unpack 4-bit values and interpret them as signed integers
            unpacked_weights = torch.zeros((qweight.shape[0]*8, qweight.shape[1]), dtype=torch.int8, device=qweight.device)
            unpacked_zeros = torch.zeros((qzeros.shape[0], qzeros.shape[1]*8), dtype=torch.int8, device=qzeros.device)

            for row in range(unpacked_weights.shape[0]):
                i = row % 8
                unpacked_weights[row, :] = (qweight[row // 8, :] >> (4 * i)) & 0xF

            for col in range(unpacked_zeros.shape[1]):
                i = col % 8
                unpacked_zeros[:, col] = (qzeros[:, col // 8] >> (4 * i)) & 0xF

            return unpacked_weights, unpacked_zeros+1

        qweight, qzeros, scales = layer.qweight, layer.qzeros, layer.scales
        # Unpack 4-bit quantized weights and zero points
        unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(qweight, qzeros)

        # Apply zero points and scales
        num_groups = scales.shape[0]                    # 32
        group_size = unpacked_qweight.shape[0] // scales.shape[0] # 128

        scales = scales.repeat_interleave(group_size, dim=0)
        unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)

        unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales

        return unpacked_qweight.T.half().detach()

    if quant_type == "8bit":
        layer_to_weight = dequantize_8bit
    elif quant_type == "4bit":
        layer_to_weight = dequantize_4bit
    else:
        raise ValueError("Invalid quant_type")
    
    for name, module in model.named_modules():
        if name.endswith("mlp") or name.endswith("self_attn"):
            for subname, submodule in module.named_children():
                if "proj" in subname:
                    target_device = submodule.weight.device
                    quantized_submodule = quantized_model.get_submodule(name + "." + subname).to(target_device)
                    dequantized_weight = layer_to_weight(quantized_submodule).to(target_device)
                    submodule.weight = nn.Parameter(dequantized_weight.to(target_device).to(submodule.weight.dtype))

                    # del dequantized_weight     
                    gc.collect()
                    torch.cuda.empty_cache()
                    # setattr(module, subname, dequantized_linear)

class LoRADiff(nn.Module):
    def __init__(self, base, finetune, rank=16, niter=20):
        super().__init__()
        diff = finetune - base
        U, S, V = torch.svd_lowrank(diff.T.float(), q=rank, niter=niter)
        self.register_buffer("base", base.T)
        self.register_parameter("A", nn.Parameter((U @ torch.diag_embed(S.sqrt())).to(base.dtype)))
        self.register_parameter("B", nn.Parameter((torch.diag_embed(S.sqrt()) @ V.T).to(base.dtype)))

    def forward(self, x):
        x = x @ self.base + (x @ self.A) @ self.B
        return x
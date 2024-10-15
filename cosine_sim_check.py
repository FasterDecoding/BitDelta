import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn
import gc
import torch.nn.functional as F
from bitdelta.diff import  save_diff, save_full_model
from bitdelta.misc import find_corr_stddev
from bitdelta.binary_gemm_kernel import pack, unpack, binary_bmm
from bitdelta.utils import get_model, parse_args, get_tokenizer
from tqdm import tqdm
from bitdelta.data import get_dataset, get_dataloader

import json
import transformers

import re
import random
import numpy as np

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(seed=0)

def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_model(model_path):
    if "mistral" in model_path or "mixtral" in model_path:
        data_type = torch.bfloat16
    else:
        data_type = torch.float16
    with torch.no_grad():
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=data_type,
            low_cpu_mem_usage=True,
            # device_map="auto"   
        ).to("cuda")
    return model




def singular_values_for_variance(tensor, variances=[0.9, 0.95]):
    """
    Calculate the minimum number of singular values needed to reach specified variance ratios.

    Parameters:
    - tensor: A 2D tensor for which to calculate the SVD.
    - variances: A list of variance ratios to calculate the minimum number of singular values for.

    Returns:
    A dictionary with the variance ratios as keys and the minimum number of singular values needed as values.
    """
    # Compute SVD
    U, S, V = torch.svd(tensor)
    # Calculate the squared singular values (proportional to variance explained)
    squared_singular_values = torch.pow(S, 2)
    total_variance = torch.sum(squared_singular_values)
    cumulative_variance_ratios = torch.cumsum(squared_singular_values, dim=0) / total_variance
    
    # Find the minimum number of singular values for each specified variance
    results = {}
    for variance in variances:
        num_singular_values = torch.searchsorted(cumulative_variance_ratios, variance) + 1  # +1 because indices start at 0
        results[variance] = num_singular_values.item()
        
    return results


def cosine_similarity_matrix(finetuned_param, pretrained_param):
    finetuned_flat = finetuned_param.view(-1)
    pretrained_flat = pretrained_param.view(-1)
    cosine_similarity = F.cosine_similarity(finetuned_flat.unsqueeze(0), pretrained_flat.unsqueeze(0), dim=1)
    return cosine_similarity.item()


def check_delta_properties(delta_weight):
    # analysis properties for each linear weight in deltas

    # 计算矩阵的Frobenius范数（二范数）
    matrix_norm = torch.norm(delta_weight, p='fro')

    # 计算矩阵的条件数 
    # 矩阵的条件数（Condition Number）衡量的是矩阵求逆的数值稳定性。具体来说，它描述了原始数据的微小变化如何影响矩阵运算的结果。条件数越高，计算结果对数据的微小变化越敏感，即数值解可能不稳定；条件数越低，矩阵和其运算则越稳定。

    # 定义
    # 对于非奇异矩阵 A，其条件数定义为矩阵 A 的范数与 A 的逆的范数的乘积：
    # 其中，范数可以是任意矩阵范数，但是最常用的是2-范数（即谱范数），此时条件数可以解释为矩阵最大奇异值与最小奇异值的比值。
    cond_number = torch.linalg.cond(delta_weight)

    # 计算矩阵的秩
    rank = torch.linalg.matrix_rank(delta_weight)

    # 计算矩阵的有效秩
    rank_eff = singular_values_for_variance(delta_weight, variances=[0.9, 0.95])
    rank_90, rank_95 = rank_eff[0.9], rank_eff[0.95]


    return matrix_norm, cond_number, rank, rank_90, rank_95




    ## First part: checkout cosine similarity in first layer FFN w1

    # if "llama" in base_model_path:
    #     #weight_key = "model.layers.0.mlp.gate_proj.weight"
    #     tensor_base = base_model.model.layers[0].mlp.gate_proj.weight
    #     tensor_ft = finetuned_model.model.layers[0].mlp.gate_proj.weight
    #     cosine_sim = F.cosine_similarity(tensor_base, tensor_ft, dim=1)
    #     overall_similarity = cosine_sim.mean()
    #     base_model_name = base_model_path.split("/")[-1]
    #     finetuned_model_name = finetuned_model_path.split("/")[-1]
    #     overall_similarity_result = overall_similarity.item()
    #     print(f"Overall Cosine Similarity between {base_model_name} and {finetuned_model_name}: {overall_similarity_result}")
    #     ## 说明是llama模型
    # elif "Mixtral" in base_model_path:
    #     tensor_base = base_model.model.layers[0].block_sparse_moe.experts[0].w1.weight
    #     tensor_ft = base_model.model.layers[0].block_sparse_moe.experts[1].w1.weight
    #     cosine_sim = F.cosine_similarity(tensor_base, tensor_ft, dim=1)
    #     overall_similarity = cosine_sim.mean()




    ## Second part: checkout delta square decline potential using scaled weight

    ## third part: checkout rank of original delta and  
    ## scaled calculation delta(relation between variance ratio and #singular values)

def analysis_delta(base_model_path, finetuned_model_path):
    pretrained_model = get_model(base_model_path)
    finetuned_model = get_model(finetuned_model_path)
    print(f"We are analysising the delta between the Pretrained model: {base_model_path} and the Finetuned model: {finetuned_model_path}")
    task_vector_param_dict = {}
    pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
    finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
    param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=[])
    
    cos_sim_list = []
    norm_list = []
    cond_number_list = []
    rank_list = []
    rank_90_list = []
    rank_95_list = []

    with torch.no_grad():
        for param_name in param_names_to_merge:
            param_list = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
            if all(char not in param_name for char in param_list):
                continue
            # import pdb
            # pdb.set_trace()
            #研究finetuned_param_dict[param_name]和pretrained_param_dict[param_name]的cosine similarity
            task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]
            #check similarity
            print(f"Investigating param_name: {param_name}")
            cos_sim = cosine_similarity_matrix(finetuned_param_dict[param_name].float(), pretrained_param_dict[param_name].float())
            cos_sim_list.append(cos_sim)
            print(f"cosine similarity between the finetuned model and pretrained model: ",cos_sim)
            #研究他们差值的统计性质
            matrix_norm, cond_number, rank, rank_90, rank_95 = check_delta_properties(task_vector_param_dict[param_name].float())
            norm_list.append(matrix_norm)
            cond_number_list.append(cond_number)
            rank_list.append(rank)
            rank_90_list.append(rank_90)
            rank_95_list.append(rank_95)
            print(f"Properties of Delta Weight---matrix_norm: {matrix_norm}, cond_number: {cond_number}, rank: {rank}, rank_90: {rank_90}, rank_95: {rank_95}")

    print(f"avg_cos_sim: {sum(cos_sim_list)/len(cos_sim_list)}")
    print(f"avg_norm: {sum(norm_list)/len(norm_list)}")
    print(f"avg_cond_number: {sum(cond_number_list)/len(cond_number_list)}")
    print(f"avg_rank: {sum(rank_list)/len(rank_list)}")
    print(f"avg_rank_90: {sum(rank_90_list)/len(rank_90_list)}")
    print(f"avg_rank_95: {sum(rank_95_list)/len(rank_95_list)}")

    print(f"Analysis end for the pretrained model: {base_model_path} and finetuned_model: {finetuned_model_path}")
    del pretrained_model
    del finetuned_model
    return

moe_base = "/home/wanghanqing/projects/models/model_ver2/Mixtral-8x7B-v0.1"
instruct_base = "/home/wanghanqing/projects/models/model_ver2/Mistral-7B-Instruct-v0.2"
base_model = "/home/wanghanqing/projects/models/model_ver2/Mistral-7B-v0.1"

code_llama13 = "/data/public/opensource_models/codellama/codellama-13b-python-hf"
wizard_coder = "/data/public/opensource_models/WizardLM/WizardCoder-Python-13B-V1.0"
llama2_7b = "/data/public/opensource_models/meta-llama/Llama-2-7b-hf"
llama2_7b_chat = "/data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf"
llama2_13b = "/data/public/opensource_models/meta-llama/Llama-2-13b-hf"
llama2_13b_chat = "/data/public/opensource_models/meta-llama/Llama-2-13b-chat-hf"
wizard_math_7b = "/data/public/opensource_models/WizardLM/WizardMath-7B-V1.0"
wizard_math_13b = "/data/public/opensource_models/WizardLM/WizardMath-13B-V1.0"
meta_math_7b = "/data/public/wangshuo/exp/ft-en-metameth-llama-2-7b/ckpts/checkpoints/epoch_2_hf"
magicoder_7b = "/data/public/wangshuo/exp/ft-en-magicoder-llama-2-7b/ckpts/checkpoints/epoch_2_hf"
magicoder_13b = "/data/public/wangshuo/exp/ft-en-magicoder-llama-2-13b/ckpts/checkpoints/epoch_2_hf"


# Mistral-7B
## base
mistral_7b = "/home/wanghanqing/projects/models/model_ver2/Mistral-7B-v0.1"
## finetuned
mistral_7b_instruct_v1 = "/home/wanghanqing/projects/models/model_ver2/Mistral-7B-Instruct-v0.1"
mistral_7b_instruct_v2 = "/home/wanghanqing/projects/models/model_ver2/Mistral-7B-Instruct-v0.2"

# llama2-7b
## base
llama2_7b = "/data/public/opensource_models/meta-llama/Llama-2-7b-hf"
## finetuned
llama2_7b_chat = "/data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf"
wizard_math_7b = "/data/public/opensource_models/WizardLM/WizardMath-7B-V1.0"
meta_math_7b = "/data/public/wangshuo/exp/ft-en-metameth-llama-2-7b/ckpts/checkpoints/epoch_2_hf"
magicoder_7b = "/data/public/wangshuo/exp/ft-en-magicoder-llama-2-7b/ckpts/checkpoints/epoch_2_hf"

# llama2-13b
## base
llama2_13b = "/data/public/opensource_models/meta-llama/Llama-2-13b-hf"
## finetuned
llama2_13b_chat = "/data/public/opensource_models/meta-llama/Llama-2-13b-chat-hf"
wizard_math_13b = "/data/public/opensource_models/WizardLM/WizardMath-13B-V1.0"
magicoder_13b = "/data/public/wangshuo/exp/ft-en-magicoder-llama-2-13b/ckpts/checkpoints/epoch_2_hf"
code_llama13 = "/data/public/opensource_models/codellama/codellama-13b-python-hf"
wizard_coder = "/data/public/opensource_models/WizardLM/WizardCoder-Python-13B-V1.0"




import sys

# 打开一个日志文件
log_file = open("analysis_log.txt", "w")

# 保存原始的标准输出
original_stdout = sys.stdout

# 重定向标准输出到文件
sys.stdout = log_file

# 你的代码，所有print函数的输出都会写入log.txt
print("This will be written to analysis_log.txt")





analysis_delta(base_model_path = llama2_7b, finetuned_model_path = llama2_7b_chat)
analysis_delta(base_model_path = llama2_7b, finetuned_model_path = wizard_math_7b)
analysis_delta(base_model_path = llama2_7b, finetuned_model_path = meta_math_7b)
analysis_delta(base_model_path = llama2_7b, finetuned_model_path = magicoder_7b)

# 恢复原始的标准输出
sys.stdout = original_stdout

# 关闭日志文件
log_file.close()
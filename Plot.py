import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
def plot_bit_delta(title):
    plt.figure(figsize=(10, 5))
    plt.plot(bit_delta, label=f'Bit-Delta {map[args.param_type]}')
    plt.plot(svd_delta, label=f'svd Data {map[args.param_type]}')
    plt.plot(mix_delta, label=f'Ours {map[args.param_type]}')
    plt.title("Comparison of the Cosine Similarity between the Bit-Delta, SVD, and our method with WizardMath-7B-v1.0")
    plt.xlabel(f'{map[args.param_type]} of each layer')  # X轴标题
    plt.ylabel('Cosine Similarity Value')  # Y轴标题
    plt.legend()
    plt.savefig(f'./figures/{map[args.param_type]}_cos_sim.pdf')   
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_type', type=str,  help='finetuned model name')
    map = {"q_proj":"Query_proj", "k_proj":"Key_proj","v_proj":"Value_proj","o_proj":"Output_proj","gate_proj":"Gate_proj","up_proj":"Up_proj","down_proj":"Down_proj"}
    args = parser.parse_args()
    
    bit_delta = torch.load(f'./statistic/{args.param_type}_bitdelta_cos_sim.pt')
    svd_delta = torch.load(f'./statistic/{args.param_type}_svd_cos_sim.pt')
    mix_delta = torch.load(f'./statistic/{args.param_type}_mix_cos_sim.pt')
    
    plot_bit_delta('Cosine Similarity of Bit-Delta, svd and mixed Data')
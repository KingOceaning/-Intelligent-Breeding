import pandas as pd
import numpy as np
import os
import subprocess

# 筛选出已经导入目标基因的样本
def sample_fliter(script_prefix: str, BCn: str, target_gene: str):
    # 读入转换后的基因型文件
    BC1_gene = pd.read_csv(script_prefix+'/BCres/%sF1/%sF1_genotypes.dat.transform'%(BCn, BCn), delimiter='\t')
    chrom, pos = list(map(int, target_gene.split('_')))
    BC1_gene_sub = BC1_gene.loc[BC1_gene['marker'].str.contains('%d_'%chrom)]
    BC1_tran = BC1_gene_sub.set_index('marker').transpose()
    samples_flited = BC1_tran.loc[BC1_tran[target_gene]==1]

    sample_info = pd.DataFrame(columns=['samples', 'Start', 'End', 'Length(left)', 'Length(right)', 'PRPG'])
    sample_info['samples'] = samples_flited.index.to_numpy()
    for Id in samples_flited.index:
        sample_ = list(samples_flited.loc[Id, :].items())
        for i in range(len(sample_)):
            idx, allele = sample_[i]
            if allele == 1 and (i == 0 or sample_[i-1][1] == 2):
                if type(idx) == type(' '):
                    idx = int(idx.split('_')[1])
                if idx <= pos:
                    sample_info.loc[sample_info['samples']==Id, 'Start'] = idx
            if allele == 1 and (i == len(sample_)-1 or sample_[i+1][1] == 2):
                if type(idx) == type(' '):
                    idx = int(idx.split('_')[1])
                if idx >= pos:
                    sample_info.loc[sample_info['samples']==Id, 'End'] = idx
                    break
    sample_info['Length(left)'] = ((pos - sample_info['Start'])/1e6).astype(float).round(2)
    sample_info['Length(right)'] = ((sample_info['End'] - pos)/1e6).astype(float).round(2)

    for s in sample_info['samples'].to_list():
        seq = BC1_gene[s].to_list()
        prpg = (seq.count(2)+len(seq)) / (2*len(seq))
        sample_info.loc[sample_info['samples'] == s, 'PRPG'] = round(prpg, 3)
    sample_info.to_csv(script_prefix + '/BCres/%sF1/%sF1.samples'%(BCn, BCn), sep='\t', index=False)

# 回报函数
def reward_compute_1(script_prefix: str):
    cost = np.sum(pd.read_csv(script_prefix + '/BCres/Cost.csv')['Quantity']).astype(int)
    return 1/cost

# 回报函数
def reward_compute_2(script_prefix: str):
    cost = np.sum(pd.read_csv(script_prefix + '/BCres/Cost.csv')['Quantity']).astype(int)
    return (60*300*4 - cost) / 50

def find_closest_values(arr, target):
    idx = np.searchsorted(arr, target)
    if idx == len(arr):
        lower = arr[-1]
        upper = np.inf
    elif idx == 0:
        lower = arr[0]
        upper = np.inf
    else:
        lower = arr[idx - 1]
        upper = arr[idx] if idx < len(arr) else np.inf
    return lower, upper

# 计算左右断开的概率
def compute_pl_pr(target_gene):
    chrom, pos_ = target_gene.split('_')
    pos_, left, right = int(pos_), int(int(pos_)-5.25e6), int(int(pos_)+5.25e6)
    gene_map = pd.read_csv('./data/10RIL_genetic_map_V3.txt', delimiter='\t')
    pro_map = pd.read_csv('./data/Corn_Genetic_Map.map', delimiter='\t').set_index('marker')
    target_p = pro_map.loc[target_gene, 'position']
    gene_map = gene_map.loc[gene_map['chr'] == int(chrom)]

    if left > 0:
        t1, t2 = find_closest_values(gene_map['pos_v3'].to_numpy(), left)
        if t2 == np.inf:
            pl = gene_map[gene_map['pos_v3'] == t1]['pos_g'].values[0]
            pl = left/t1 * pl
        else:
            l, r = gene_map[gene_map['pos_v3'] == t1]['pos_g'].values[0], gene_map[gene_map['pos_v3'] == t2]['pos_g'].values[0]
            pl = ((left-t1)/(t2-t1) * (r-l) + l)
    t1, t2 = find_closest_values(gene_map['pos_v3'].to_numpy(), right)
    if t2 == np.inf:
        pr = gene_map[gene_map['pos_v3'] == t1]['pos_g'].values[0]
        pr = right/t1 * pr
    else:
        l, r = gene_map[gene_map['pos_v3'] == t1]['pos_g'].values[0], gene_map[gene_map['pos_v3'] == t2]['pos_g'].values[0]
        pr = ((right-t1)/(t2-t1) * (r-l) + l)
    pl = (target_p - pl) / 100
    pr = (pr - target_p) / 100
    return pl, pr
import os
import pandas as pd
import numpy as np
import subprocess

'''
Z658M_1_4为非轮回亲本P1
Z658F_4_3 为轮回亲本P2
'''

# 模拟BCn结果
def simulate_BCn(P: list, BCn: str, BC_nums: list, script_prefix: str):

    # script_prefix = '/home/lizy/crop/Simulate_Env'
    chrom_file = script_prefix + '/data/Corn_Genetic_Map.chrom'
    ped_file = script_prefix + '/data/%sF1/%sF1.ped'%(BCn, BCn)
    gen_file = script_prefix + '/data/%sF1/%sF1.gen'%(BCn, BCn)
    map_file = script_prefix + '/data/Corn_Genetic_Map.map'
    out_path = script_prefix + '/BCres/%sF1/%sF1'%(BCn, BCn)

    os.makedirs(script_prefix + '/data/%sF1'%(BCn), exist_ok=True)
    os.makedirs(script_prefix + '/BCres/%sF1'%(BCn), exist_ok=True)

    # 生成.ped文件，记录每一个体的亲本
    with open(script_prefix + '/data/%sF1/%sF1.ped'%(BCn, BCn), 'w') as ped:
        sum_num = np.sum(BC_nums)
        ped.write('Name\tParent1\tParent2\n')
        ped.write('\t'.join(['P2', 'NA', 'NA']) + '\n')
        for p in P:
            ped.write('\t'.join([p, 'NA', 'NA']) + '\n')
        j = 1
        for idx, BC_num in enumerate(BC_nums):
            for i in range(j, j+BC_num):
                ped.write('\t'.join(['%sF1_%d'%(BCn, i), P[idx], 'P2']) + '\n')
            j += BC_num

    # 生成.gen文件，记录亲本每一个位点上的等位基因的基因型
    gen = pd.DataFrame(columns=['marker', 'P2_1', 'P2_2'])
    Map = pd.read_csv(script_prefix + '/data/Corn_Genetic_Map.map', delimiter='\t')
    parent_gen = pd.read_csv(script_prefix + '/BCres/%s/%s_genotypes.dat'%(P[0].split('_')[0], P[0].split('_')[0]), delimiter='\t')
    gen['marker'] = Map['marker']
    gen['P2_1'] = np.array([2] * gen['marker'].shape[0]).astype(int)
    gen['P2_2'] = np.array([2] * gen['marker'].shape[0]).astype(int)
    for p in P:
        parent_gen.loc[:, parent_gen.columns.str.contains(p+'_')]
        gen = pd.concat((gen, parent_gen.loc[:, parent_gen.columns.str.contains(p+'_')]), axis=1)
    gen.to_csv(script_prefix + '/data/%sF1/%sF1.gen'%(BCn, BCn), sep='\t', index=False)

    # 生成.par文件，记录pedigree运行需要的参数
    with open(script_prefix + '/data/%sF1/%sF1.par'%(BCn, BCn), 'w') as par_file:
        par_info = f"""
PLOIDY = 2
MAPFUNCTION = HALDANE 
MISSING = NA
CHROMFILE = {chrom_file}
PEDFILE = {ped_file}
MAPFILE = {map_file}
FOUNDERFILE = {gen_file} 
OUTPUT = {out_path}
ALLOWNOCHIASMATA = 0
NATURALPAIRING = 1 
PARALLELQUADRIVALENTS = 0.0
PAIREDCENTROMERES = 0.0
SEED = {str(np.random.randint(100000))}
"""
        par_file.write(par_info)

    # 模拟BCn代
    subprocess.run(['java', '-jar', script_prefix + '/tools/PedigreeSim.jar', script_prefix + '/data/%sF1/%sF1.par'%(BCn, BCn)], stdout=subprocess.PIPE)

    # 只保留基因型文件
    f_list = os.listdir(script_prefix + '/BCres/%sF1'%BCn)
    f_list.remove('%sF1_genotypes.dat'%BCn)
    for f in f_list:
        os.remove(script_prefix + '/BCres/%sF1/'%BCn + f)
    
    # 转换基因型文件
    df_org = pd.read_csv(script_prefix + '/BCres/%sF1/%sF1_genotypes.dat'%(BCn, BCn), delimiter='\t')
    df_tra = pd.DataFrame(columns=['marker'] + ['%sF1_%d'%(BCn, i) for i in range(1, np.sum(BC_nums)+1)])
    df_tra['marker'] = df_org['marker']
    for i in range(1, np.sum(BC_nums) + 1):
        chr_1, chr_2 = df_org['%sF1_%d_1'%(BCn, i)].to_numpy(dtype=int), df_org['%sF1_%d_2'%(BCn, i)].to_numpy(dtype=int)
        tras = (chr_1 + chr_2) / 2
        df_tra.loc[:, '%sF1_%d'%(BCn, i)] = tras.astype(int)
    df_tra.to_csv(script_prefix + '/BCres/%sF1/%sF1_genotypes.dat.transform'%(BCn, BCn), sep='\t', float_format='%1.f', index=False)

    # 记录BCn代个体数量
    df = pd.read_csv(script_prefix + '/BCres/Cost.csv').set_index('Generation')
    df.loc[BCn, 'Quantity'] = np.sum(BC_nums)
    df.to_csv(script_prefix + '/BCres/Cost.csv')
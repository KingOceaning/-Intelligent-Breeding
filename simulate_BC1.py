import os
import pandas as pd
import numpy as np
import subprocess

'''
Z658M_1_4为非轮回亲本P1
Z658F_4_3 为轮回亲本P2
'''

# 模拟BC1代结果
def simulate_BC1(BC1_num, script_prefix):
    # script_prefix = '/home/lizy/crop/Simulate_Env'

    # 生成.ped文件，记录每一个体的亲本
    with open(script_prefix + '/data/BC1F1/BC1F1.ped', 'w') as ped_file:
        ped_file.write('Name\tParent1\tParent2\n')
        ped_file.write("P1\tNA\tNA\nP2\tNA\tNA\nF1\tP1\tP2\n")
        for i in range(1, BC1_num+1):
            ped_file.write('BC1F1_%d\tP2\tF1\n'%(i))
    
    # 生成.gen文件，记录亲本每一个位点上的等位基因的基因型
    with open(script_prefix + '/data/BC1F1/BC1F1.gen', 'w') as gen_file:
        gen_file.write('marker\tP1_1\tP1_2\tP2_1\tP2_2\n')
        with open('./data/Corn_Genetic_Map.map', 'r') as map_file:
            for line in map_file.readlines()[1:]:
                marker = line.strip().split('\t')[0]
                gen_file.write('%s\t0\t0\t2\t2\n'%marker)
    
    chrom_file = script_prefix + '/data/Corn_Genetic_Map.chrom'
    ped_file = script_prefix + '/data/BC1F1/BC1F1.ped'
    gen_file = script_prefix + '/data/BC1F1/BC1F1.gen'
    map_file = script_prefix + '/data/Corn_Genetic_Map.map'
    out_path = script_prefix + '/BCres/BC1F1/BC1F1'
    # 生成.par文件，记录pedigree运行需要的参数
    with open(script_prefix + '/data/BC1F1/BC1F1.par', 'w') as par_file:
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
    
    # 模拟BC1代回交结果
    subprocess.run(['java', '-jar', script_prefix + '/tools/PedigreeSim.jar', script_prefix + '/data/BC1F1/BC1F1.par'], stdout=subprocess.PIPE)

    # 只保留基因型文件
    f_list = os.listdir(script_prefix + '/BCres/BC1F1')
    f_list.remove('BC1F1_genotypes.dat')
    for f in f_list:
        os.remove(script_prefix + '/BCres/BC1F1/' + f)
    
    # 转换基因型文件
    df_org = pd.read_csv(script_prefix + '/BCres/BC1F1/BC1F1_genotypes.dat', delimiter='\t')
    df_tra = pd.DataFrame(columns=['marker'] + ['BC1F1_%d'%i for i in range(1, BC1_num+1)])
    df_tra['marker'] = df_org['marker']
    for i in range(1, BC1_num + 1):
        chr_1, chr_2 = df_org['BC1F1_%d_1'%i].to_numpy(dtype=int), df_org['BC1F1_%d_2'%i].to_numpy(dtype=int)
        tras = (chr_1 + chr_2) / 2
        df_tra.loc[:, 'BC1F1_%d'%i] = tras.astype(int)
    df_tra.to_csv(script_prefix + '/BCres/BC1F1/BC1F1_genotypes.dat.transform', sep='\t', float_format='%1.f', index=False)

    # 记录BC1代个体数量
    df = pd.DataFrame(columns=['Generation', 'Quantity']).set_index('Generation')
    df.loc['BC1', 'Quantity'] = BC1_num
    df.to_csv(script_prefix + '/BCres/Cost.csv')
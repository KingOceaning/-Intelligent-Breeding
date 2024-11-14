from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import pickle
import json
from simulate_BC1 import simulate_BC1
from simulate_BCn import simulate_BCn
from sample_select import *

class RF_for_BC:
    def __init__(self):
        self.Max_Generation = 5
        self.RFs = [RandomForestRegressor(n_estimators=150, random_state=0) for i in range(5)]
        

    def train(self, BCn:int, test_size=0.05):
        data = pd.read_csv('./memory/BC%d.csv'%(BCn))
        X, y = data.iloc[:, :-1].to_numpy(), data.iloc[:, -1].to_numpy()
        # print(X.shape, y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=test_size) 
        self.RFs[BCn-1].fit(X_train, y_train)

    def save_model(self, prefix):
        for i in range(self.Max_Generation):
            with open(prefix + 'RF_for_BC%d.pkl'%(i+1), 'wb') as f:
                pickle.dump(self.RFs[i], f)

    def load_model(self, prefix):
        for i in range(self.Max_Generation):
            with open(prefix + '/RF_for_BC%d.pkl'%(i+1), 'rb') as f:
                self.RFs[i] = pickle.load(f)
    
    # 迭代更新模型
    def iteration_one_time(self, target_gene):
        script_prefix = '/home/lizy/crop/Simulate_Env/script'
        pl, pr = compute_pl_pr(target_gene)
        record = pd.DataFrame(columns=['PRPG', 'Nt', 'pl', 'pr', 'nums', 'bc_nums', 'reward'])
        total_reward = 6*60*300
        # 模拟BC1决策
        bc_num = 1000
        simulate_BC1(bc_num, script_prefix)
        total_reward -= 1000
        sample_fliter(script_prefix, 'BC1', target_gene)
        samples = pd.read_csv('./BCres/BC1F1/BC1F1.samples', delimiter='\t').sort_values(by='PRPG', ascending=False)
        samples.loc[samples['Length(left)'] < 5.25, 'pl'] = 1.0
        samples.loc[samples['Length(left)'] >= 5.25, 'pl'] = pl
        samples.loc[samples['Length(right)'] < 5.25, 'pr'] = 1.0
        samples.loc[samples['Length(right)'] >= 5.25, 'pr'] = pr
        Nt_df = samples.loc[(samples['Length(left)'] < 5.25) & (samples['Length(right)'] < 5.25)]
        Nt = Nt_df.shape[0]
        rec = [np.mean(samples['PRPG']), Nt, np.mean(samples['pl']), np.mean(samples['pr'])]
        act = self.decision(0, rec)
        record.loc[0] = rec + act + [0.0]
        print('BC1: ', rec[:4], ' action: ', act)
        
        # 模拟2~6代回交决策
        for i in range(2, 6):
            P = list(set(samples.head(act[0])['samples'].to_list()) | set(Nt_df['samples'].to_list()))[:act[0]]
            simulate_BCn(P, 'BC%d'%i, [act[1]]*act[0], script_prefix)
            total_reward -= act[0] * act[1]
            sample_fliter(script_prefix, 'BC%d'%i, target_gene)
            samples = pd.read_csv('./BCres/BC%sF1/BC%sF1.samples'%(i, i), delimiter='\t')
            samples.loc[samples['Length(left)'] < 5.25, 'pl'] = 1.0
            samples.loc[samples['Length(left)'] >= 5.25, 'pl'] = pl
            samples.loc[samples['Length(right)'] < 5.25, 'pr'] = 1.0
            samples.loc[samples['Length(right)'] >= 5.25, 'pr'] = pr
            Nt_df = samples.loc[(samples['Length(left)'] < 5.25) & (samples['Length(right)'] < 5.25)]
            Nt = Nt_df.shape[0]
            rec = [np.mean(samples['PRPG']), Nt, np.mean(samples['pl']), np.mean(samples['pr'])]
            act = self.decision(i-1, rec)
            record.loc[i-1] = (rec + act + [0.0]).copy()
            print(f'BC{i}: ', rec[:4], ' action: ', act)
            # 终止条件
            if i > 3 and Nt_df.loc[Nt_df['PRPG'] > 0.985].shape[0] >= 5:
                break
            elif i == 5 and (Nt_df.loc[Nt_df['PRPG'] > 0.985].shape[0] < 5) :
                i += 1
                P = list(set(samples.head(act[0])['samples'].to_list()) | set(Nt_df['samples'].to_list()))[:act[0]]
                simulate_BCn(P, 'BC%d'%i, [act[1]]*act[0], script_prefix)
                total_reward -= act[0] * act[1]
                sample_fliter(script_prefix, 'BC%d'%i, target_gene)
                samples = pd.read_csv('./BCres/BC%sF1/BC%sF1.samples'%(i, i), delimiter='\t')
                samples.loc[samples['Length(left)'] < 5.25, 'pl'] = 1.0
                samples.loc[samples['Length(left)'] >= 5.25, 'pl'] = pl
                samples.loc[samples['Length(right)'] < 5.25, 'pr'] = 1.0
                samples.loc[samples['Length(right)'] >= 5.25, 'pr'] = pr
                Nt_df = samples.loc[(samples['Length(left)'] < 5.25) & (samples['Length(right)'] < 5.25)]
                Nt = Nt_df.shape[0]
                if (Nt_df.loc[Nt_df['PRPG'] > 0.985].shape[0] < 5):
                    total_reward = 0.0
        # 计算total reward
        total_reward = total_reward / 50
        record.loc[:, 'reward'] = total_reward
        # 迭代训练各代的模型 
        for i in range(record.shape[0]-1):  # -1是为了排除掉最后一行记录，因为最后一行并没有执行action
            # print(X[i].reshape(1, -1), np.array([y[i]]))
            memory = pd.read_csv('./memory/BC%d.csv'%(i+1))
            memory.to_csv('./memory/BC%d.csv'%(i+1), index=False)
            memory.loc[memory.shape[0], :] = record.loc[i, :].to_numpy()
            self.RFs[i].fit(memory.iloc[:, :-1].to_numpy(), memory.iloc[:, -1].to_numpy())

    def generation_decision(self, target_gene):
        script_prefix = '/home/lizy/crop/Simulate_Env/script'
        pl, pr = compute_pl_pr(target_gene)
        seq = []
        total_reward = 6*60*300
        # 模拟BC1
        bc_num = 1000
        simulate_BC1(bc_num, script_prefix)
        total_reward -= 1000
        sample_fliter(script_prefix, 'BC1', target_gene)
        samples = pd.read_csv('./BCres/BC1F1/BC1F1.samples', delimiter='\t').sort_values(by='PRPG', ascending=False)
        samples.loc[samples['Length(left)'] < 5.25, 'pl'] = 1.0
        samples.loc[samples['Length(left)'] >= 5.25, 'pl'] = pl
        samples.loc[samples['Length(right)'] < 5.25, 'pr'] = 1.0
        samples.loc[samples['Length(right)'] >= 5.25, 'pr'] = pr
        Nt_df = samples.loc[(samples['Length(left)'] < 5.25) & (samples['Length(right)'] < 5.25)]
        Nt = Nt_df.shape[0]
        rec = [np.mean(samples['PRPG']), Nt, np.mean(samples['pl']), np.mean(samples['pr'])]
        act = self.decision(0, rec)
        seq.append({"State":rec.copy(), "Action": act.copy(), "Reward":0.0})
        print('BC1: ', rec[:4], ' action: ', act)
        
        # 模拟2~6代
        for i in range(2, 6):
            P = list(set(samples.head(act[0])['samples'].to_list()) | set(Nt_df['samples'].to_list()))[:act[0]]
            simulate_BCn(P, 'BC%d'%i, [act[1]]*act[0], script_prefix)
            total_reward -= act[0] * act[1]
            sample_fliter(script_prefix, 'BC%d'%i, target_gene)
            samples = pd.read_csv('./BCres/BC%sF1/BC%sF1.samples'%(i, i), delimiter='\t')
            samples.loc[samples['Length(left)'] < 5.25, 'pl'] = 1.0
            samples.loc[samples['Length(left)'] >= 5.25, 'pl'] = pl
            samples.loc[samples['Length(right)'] < 5.25, 'pr'] = 1.0
            samples.loc[samples['Length(right)'] >= 5.25, 'pr'] = pr
            Nt_df = samples.loc[(samples['Length(left)'] < 5.25) & (samples['Length(right)'] < 5.25)]
            Nt = Nt_df.shape[0]
            rec = [np.mean(samples['PRPG']), Nt, np.mean(samples['pl']), np.mean(samples['pr'])]
            act = self.decision(i-1, rec)
            seq.append({"State":rec.copy(), "Action": act.copy(), "Reward":0.0})
            print(f'BC{i}: ', rec[:4], ' action: ', act)
            # 终止条件
            if i > 3 and Nt_df.loc[Nt_df['PRPG'] > 0.985].shape[0] >= 5:
                break
            elif i == 5 and (Nt_df.loc[Nt_df['PRPG'] > 0.985].shape[0] < 5):
                i += 1
                P = list(set(samples.head(act[0])['samples'].to_list()) | set(Nt_df['samples'].to_list()))[:act[0]]
                simulate_BCn(P, 'BC%d'%i, [act[1]]*act[0], script_prefix)
                total_reward -= act[0] * act[1]
                sample_fliter(script_prefix, 'BC%d'%i, target_gene)
                samples = pd.read_csv('./BCres/BC%sF1/BC%sF1.samples'%(i, i), delimiter='\t')
                samples.loc[samples['Length(left)'] < 5.25, 'pl'] = 1.0
                samples.loc[samples['Length(left)'] >= 5.25, 'pl'] = pl
                samples.loc[samples['Length(right)'] < 5.25, 'pr'] = 1.0
                samples.loc[samples['Length(right)'] >= 5.25, 'pr'] = pr
                Nt_df = samples.loc[(samples['Length(left)'] < 5.25) & (samples['Length(right)'] < 5.25)]
                Nt = Nt_df.shape[0]
                if (Nt_df.loc[Nt_df['PRPG'] > 0.985].shape[0] < 5):
                    total_reward = 0.0
        # 计算total reward
        total_reward = total_reward / 50
        print('\nReward: ', total_reward)
        for i in range(len(seq) - 1):  # -1是为了排除掉最后一行记录，因为最后一行并没有执行action
            seq[i]['Reward'] = total_reward
        return seq


    # 生成每一代的回交策略
    def decision(self, BCn, state): # BCn：当前需要决策的代数[0，6)，state：状态空间的表示向量
        budget = -np.inf
        action = [60, 300]
        for num in range(20, 61, 2):
            for bc_nums in range(100, 350, 50):
                reward = self.RFs[BCn].predict(np.array(state + [num, bc_nums]).reshape(1, -1))[0]
                # print("Action:", [num, bc_nums], "Reward:", reward)
                if reward > budget:
                    budget = reward
                    action = [num, bc_nums]
        return action

    # 重置记忆的方法
    # PS：别动sequence文件夹里的json文件，要不然就要重新生成(>_<)
    def reset_memory(self):
        for i in range(1, 6):
            fl = os.listdir('./sequence')
            BC_frame = [pd.DataFrame(columns=['prpg','Nt', 'Pl', 'Pr', 'nums', 'bc_nums', 'Reward']) for i in range(6)]
            for fn in fl:
                with open('./sequence/' + fn, 'r', encoding='utf-8') as f:
                    sequence = json.load(f)
                    for i in range(100):
                        for j in range(len(sequence[i])):
                            col = (sequence[i][j]['State'] + sequence[i][j]['Action'])
                            col.append(sequence[i][j]['Reward'])
                            BC_frame[j].loc[BC_frame[j].shape[0]] = col.copy()
        for i in range(6):
            BC_frame[i].to_csv(f'./memory/BC{i+1}.csv', index=False)
    

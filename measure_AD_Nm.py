import numpy as np
import torch as tc
import Parameters
import math
import copy
import pickle
from collections import Counter
import random


class measure_Nm_AD:
    def __init__(self, para=Parameters.measure_data()):
        self.para = copy.deepcopy(para)
        self.measure_mode_init = self.para['measure_mode_init']
        if self.measure_mode_init:
            self.prepare_initialize()

    def prepare_initialize(self):
        f = open(self.para['load_state_path'], 'rb')  # 加载目标态
        self.lps0 = pickle.load(f)
        f.close()
        for k in range(self.para['retained_feature']):
            self.lps0[k] = tc.from_numpy(self.lps0[k]).to(self.para['dtype']).to(self.para['device'])
        self.eigvector = list(range(3))  # 测量基底
        self.eigvector[0] = [tc.tensor([1 / math.sqrt(2), 1 / math.sqrt(2)], dtype=self.para['dtype'], device=self.para['device']),
                             tc.tensor([1 / math.sqrt(2), -1 / math.sqrt(2)], dtype=self.para['dtype'], device=self.para['device'])]
        self.eigvector[1] = [tc.tensor([1 / math.sqrt(2), 1j / math.sqrt(2)], dtype=self.para['dtype'], device=self.para['device']),
                             tc.tensor([1 / math.sqrt(2), -1j / math.sqrt(2)], dtype=self.para['dtype'], device=self.para['device'])]
        self.eigvector[2] = [tc.tensor([1, 0], dtype=self.para['dtype'], device=self.para['device']),
                             tc.tensor([0, 1], dtype=self.para['dtype'], device=self.para['device'])]
        self.population = list()  # 编码2^N种测量结果
        for pp in range(2 ** self.para['retained_feature']):
            element = bin(pp)[2:]
            element = (self.para['retained_feature'] - len(element)) * '0' + element
            self.population.append(element)
        self.base_List = np.array([[0, 1], [2, 3], [4, 5]])  # 编码基底

    def initialize_document_new(self):
        if self.measure_mode_init == 'True':
            f = open(self.para['save_Nmdata_path'] + '%d_%d' % (self.measure_num, self.para['sample_num']) + '.txt', 'w+')
            f.close()
        else:
            # 打开旧文件
            with open(self.para['save_Nmdata_path'] + '%d_%d' % ((self.measure_num-self.para['measure_step']), self.para['sample_num']) + '.txt', 'r') as source:
                lines = source.readlines()
                lines = lines[1:]
            # 创建新文件并写入旧文件的内容
            with open(self.para['save_Nmdata_path'] + '%d_%d' % (self.measure_num, self.para['sample_num']) + '.txt', 'w+') as new:
                new.writelines(lines)

    def calculate_probability(self, lps, nm, ns):
        tmp_l = list(range(self.para['retained_feature']))
        for ii in range(0, self.para['retained_feature']):
            tmp_l[ii] = tc.einsum('abcd,c->abd', [lps[ii], self.eigvector[nm[ii]][ns[ii]].conj()])
        tmp_e = tc.einsum('abc,ade->bcde', [tmp_l[0], tmp_l[0].conj()])
        for ii in range(1, self.para['retained_feature']):
            tmp = tc.einsum('abcd,ebf->afcde', [tmp_e, tmp_l[ii]])
            tmp_e = tc.einsum('afcde,edg->afcg', [tmp, tmp_l[ii].conj()])
        p = tc.squeeze(tmp_e).real.to(tc.float32)
        return p

    '''train dataset'''
    def start_measure(self):
        array_save = list(range(self.para['retained_feature']))
        if self.measure_mode_init == 'True':
            self.measure_num = self.para['measure_num_init']
            iteration_num = self.para['measure_num_init']
        else:
            self.measure_num += self.para['measure_step']
            iteration_num = self.para['measure_step']
        self.initialize_document_new()
        print('\n&&&&&&&& measure_num: %d &&&&&&&&' % (self.measure_num))
        number_tot = 0
        for i in range(iteration_num):
            if self.measure_mode_init == 'True':
                print('\n-------- measure_train: %d --------' % (i))
            else:
                print('\n-------- measure_train: %d --------' % (self.measure_num - self.para['measure_step'] + i))
            m = np.random.randint(3, size=self.para['retained_feature'])  # select 测量算符
            #  Random initial
            weight = []
            for j in range(len(self.population)):
                s = list(range(self.para['retained_feature']))
                for k in range(self.para['retained_feature']):
                    s[k] = int(self.population[j][k])
                p = self.calculate_probability(self.lps0, m, s)  # calculate the probability
                weight.append(p)
            res = Counter(random.choices(self.population, weight, k=self.para['sample_num']))
            number_tot += len(res.keys())
            print(number_tot)

            for key in res.keys():
                s = list(range(self.para['retained_feature']))
                for k in range(self.para['retained_feature']):
                    s[k] = int(key[k])
                for u in range(self.para['retained_feature']):
                    array_save[u] = self.base_List[m[u]][s[u]]
                f = open(self.para['save_Nmdata_path'] + '%d_%d' % (self.measure_num, self.para['sample_num']) + '.txt', 'a+')
                save_str = ", ".join(map(str, array_save[:self.para['retained_feature']]))
                normalized_res = res[key] / self.para['sample_num']
                f.write(f"{save_str}, {normalized_res:.16f}\n")
                f.close()
            # add count_tot in line1
        with open(self.para['save_Nmdata_path'] + '%d_%d' % (self.measure_num, self.para['sample_num']) + '.txt', 'r+') as f:
            lines = f.readlines()
            line_count = len(lines)
            f.seek(0)  # 回到文件开头
            lines.insert(0, str(line_count) + '\n')  # 在第一行插入行数
            f.writelines(lines)
        self.measure_mode_init = 'False'
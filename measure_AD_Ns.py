import numpy as np
import Parameters
import copy
from collections import Counter
import random
import measure_AD_Nm


class Sampling_Ns_AD(measure_AD_Nm.measure_Nm_AD):
    def __init__(self, para=Parameters.sample_data()):
        measure_AD_Nm.measure_Nm_AD.__init__(self, para=Parameters.measure_data())
        self.para = copy.deepcopy(para)
        self.sample_mode_init = self.para['sample_mode_init']
        if self.sample_mode_init:
            self.prepare_initialize()
        self.measure_Nm = dict()  # 测量、采样初始化
        for k in range(self.para['measure_train']):
            self.measure_Nm['%d' % k] = np.random.randint(3, size=self.para['retained_feature'])

    def initialize_document(self):
        f = open(self.para['save_Nsdata_path'] + '%d_%d' % (self.para['measure_train'], self.sample_num) + '.txt', 'w+')
        f.close()

    '''train dataset'''
    def start_sample(self):
        array_save = list(range(self.para['retained_feature']))
        if self.sample_mode_init == 'True':
            self.sample_num = self.para['sample_num_init']
        else:
            self.sample_num += self.para['sample_step']
        print('\n&&&&&&&& measure_num: %d &&&&&&&&' % (self.para['measure_train']))
        self.initialize_document()
        number_tot = 0
        for i in range(self.para['measure_train']):  # 测量算符数
            print('\n-------- measure_train: %d --------' % (i))
            #  Random initial
            m = self.measure_Nm['%d' % i]  # select 测量算符
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
                s1 = list(range(self.para['retained_feature']))
                for k in range(self.para['retained_feature']):
                    s1[k] = int(key[k])
                for u in range(self.para['retained_feature']):
                    array_save[u] = self.base_List[m[u]][s1[u]]
                f = open(self.para['save_Nsdata_path'] + '%d_%d' % (self.para['measure_train'], self.sample_num) + '.txt', 'a+')
                save_str = ", ".join(map(str, array_save[:self.para['retained_feature']]))
                normalized_res = res[key] / self.para['sample_num']
                f.write(f"{save_str}, {normalized_res:.16f}\n")
                f.close()
            # add count_tot in line1
        with open(self.para['save_Nsdata_path'] + '%d_%d' % (self.para['measure_train'], self.sample_num) + '.txt',
                  'r+') as f:
            lines = f.readlines()
            line_count = len(lines)
            f.seek(0)  # 回到文件开头
            lines.insert(0, str(line_count) + '\n')  # 在第一行插入行数
            f.writelines(lines)
        self.sample_mode_init = 'False'


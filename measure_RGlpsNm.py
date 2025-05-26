import numpy as np
import torch as tc
import Parameters
import math
import copy
import pickle


class measure_Nm:
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
        for i in range(iteration_num):
            if self.measure_mode_init == 'True':
                print('\n-------- measure_train: %d --------' % (i))
            else:
                print('\n-------- measure_train: %d --------' % (self.measure_num - self.para['measure_step'] + i))
            count_tmp = 0
            m = np.random.randint(3, size=self.para['retained_feature'])  # select 测量算符
            #  Random initial
            s = np.random.randint(2, size=self.para['retained_feature'])
            y0 = self.calculate_probability(self.lps0, m, s)  # calculate the probability
            #  Important sampling
            for j in range(2000000):
                walking = np.random.randint(-1, 2, size=self.para['retained_feature'])
                s1 = s + walking
                for jj in range(self.para['retained_feature']):
                    if -1 < s1[jj] < 2:
                        s[jj] = s1[jj]
                    else:
                        s[jj] = s[jj]
                y1 = self.calculate_probability(self.lps0, m, s)  # calculate the probability

                for k in range(len(array_save)):
                    array_save[k] = self.base_List[m[k]][s[k]]

                if y0 < y1:
                    y0 = y1
                    count_tmp += 1
                    f = open(self.para['save_Nmdata_path'] + '%d_%d' % (self.measure_num, self.para['sample_num']) + '.txt', 'a+')
                    f.write(", ".join(map(str, array_save[:self.para['retained_feature']])) + "\n")
                    f.close()
                elif np.random.uniform(0, 1) < (y1 / y0):
                    y0 = y0
                    count_tmp += 1
                    f = open(self.para['save_Nmdata_path'] + '%d_%d' % (self.measure_num, self.para['sample_num']) + '.txt', 'a+')
                    f.write(", ".join(map(str, array_save[:self.para['retained_feature']])) + "\n")
                    f.close()
                if count_tmp == self.para['sample_num']:
                    break
            print(count_tmp)
            # add count_tot in line1
        with open(self.para['save_Nmdata_path'] + '%d_%d' % (self.measure_num, self.para['sample_num']) + '.txt', 'r+') as f:
            lines = f.readlines()
            line_count = len(lines)
            f.seek(0)  # 回到文件开头
            lines.insert(0, str(line_count) + '\n')  # 在第一行插入行数
            f.writelines(lines)
        self.measure_mode_init = 'False'
import numpy as np
import Parameters
import copy
import measure_RGlpsNm


class Sampling_Ns(measure_RGlpsNm.measure_Nm):
    def __init__(self, para=Parameters.sample_data()):
        measure_RGlpsNm.measure_Nm.__init__(self, para=Parameters.measure_data())
        self.para = copy.deepcopy(para)
        self.sample_mode_init = self.para['sample_mode_init']
        if self.sample_mode_init:
            self.prepare_initialize()
        self.measure_Nm = dict()  # 测量、采样初始化
        self.Ns_init = dict()
        for k in range(self.para['measure_train']):
            self.measure_Nm['%d' % k] = np.random.randint(3, size=self.para['retained_feature'])
            self.Ns_init['%d' % k] = np.random.randint(2, size=self.para['retained_feature'])

    def initialize_document(self):
        if self.sample_mode_init == 'True':
            f = open(self.para['save_Nsdata_path'] + '%d_%d' % (self.para['measure_train'], self.sample_num) + '.txt', 'w+')
            f.close()
        else:
            # 打开旧文件
            with open(self.para['save_Nsdata_path'] + '%d_%d' % (self.para['measure_train'], (self.sample_num-self.para['sample_step'])) + '.txt', 'r') as source:
                lines = source.readlines()
                lines = lines[1:]
            # 创建新文件并写入旧文件的内容
            with open(self.para['save_Nsdata_path'] + '%d_%d' % (self.para['measure_train'], self.sample_num) + '.txt', 'w+') as new:
                new.writelines(lines)

    '''train dataset'''
    def start_sample(self, ns_init):
        array_save = list(range(self.para['retained_feature']))
        if self.sample_mode_init == 'True':
            self.sample_num = self.para['sample_num_init']
            iteration_num = self.sample_num
        else:
            self.sample_num += self.para['sample_step']
            iteration_num = self.para['sample_step']
        print('\n&&&&&&&& measure_num: %d &&&&&&&&' % (self.para['measure_train']))
        self.initialize_document()
        for i in range(self.para['measure_train']):  # 测量算符数
            print('\n-------- measure_train: %d --------' % (i))
            #  Random initial
            m = self.measure_Nm['%d' % i]  # select 测量算符
            s = ns_init['%d' % i]  # select 测量结果
            count_tmp = 0
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
                y1 = self.calculate_probability(self.lps0, m, s)

                for k in range(len(array_save)):
                    array_save[k] = self.base_List[m[k]][s[k]]

                if y0 < y1:
                    y0 = y1
                    count_tmp += 1
                    if count_tmp <= iteration_num:
                        f = open(self.para['save_Nsdata_path']+'%d_%d' % (self.para['measure_train'], self.sample_num) + '.txt', 'a+')
                        f.write(", ".join(map(str, array_save[:self.para['retained_feature']])) + "\n")
                        f.close()
                elif np.random.uniform(0, 1) < (y1 / y0):
                    y0 = y0
                    count_tmp += 1
                    if count_tmp <= iteration_num:
                        f = open(self.para['save_Nsdata_path']+'%d_%d' % (self.para['measure_train'], self.sample_num) + '.txt', 'a+')
                        f.write(", ".join(map(str, array_save[:self.para['retained_feature']])) + "\n")
                        f.close()
                if count_tmp == iteration_num:
                    break
            self.Ns_init['%d' % i] = s
            print(count_tmp)
        # add count_tot in line1
        with open(self.para['save_Nsdata_path']+'%d_%d' % (self.para['measure_train'], self.sample_num) + '.txt', 'r+') as f:
            lines = f.readlines()
            line_count = len(lines)
            f.seek(0)  # 回到文件开头
            lines.insert(0, str(line_count) + '\n')  # 在第一行插入行数
            f.writelines(lines)
        self.sample_mode_init = 'False'
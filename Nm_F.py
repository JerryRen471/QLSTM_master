import numpy as np
import os
import matplotlib.pyplot as plt
import json

para = dict()
para['epoch'] = 100
para['num_f'] = 10
para['chi'] = 16
para['measure_train'] = 100
para['sample_num'] = 10000
para['hx'] = 1
chain_dir = f"chain{para['num_f']}"
result_path = os.path.join('Result', 'Rand_large', chain_dir, 'normal_1_3_0')

def extract_from_json(path):
    with open(path, mode='r') as json_file:
        dic = json.load(json_file)
    loss_ = [float(i) for i in dic['trainloss_List']]
    loss_log_ = [float(i) for i in dic['trainloss_log']]
    fide_ = [float(i) for i in dic['fide1']]
    loss = np.mean(loss_[-10: -1])
    loss_log = np.mean(loss_log_[-10: -1])
    fide = np.mean(fide_[-10: -1])
    return loss, loss_log, fide

Nm_list = [10 + 10*i for i in range(10)]
print(Nm_list)
Ns = 10000

loss_ls = []
loss_log_ls = []
in_fide_ls = []
fide_ls = []
for Nm in Nm_list:
    path = os.path.join(result_path, f'{Nm}_{Ns}_chi{para["chi"]}_miu{230}.json')
    loss, loss_log, fide = extract_from_json(path)
    loss_ls.append(loss)
    loss_log_ls.append(loss_log)
    fide_ls.append(fide)
    in_fide_ls.append(1 - fide)

plt.figure()
x = [(i) for i in Nm_list]
y = fide_ls
plt.ylim(min(y), 1)
plt.scatter(x[1:], y[1:])
plt.xlabel('N_m')
plt.ylabel('F')
plt.savefig(f'Picture/Fide_Vs_Nm.svg')

plt.figure()
plt.title('Ns={}'.format(10000))
x = [np.log10(i) for i in Nm_list]
y = [np.log10(i) for i in in_fide_ls]
# plt.ylim(0, max(y)*1.1)
plt.scatter(x[1:], y[1:])
plt.xlabel('log(N_m)')
plt.ylabel('log(F_in)')
# Linear regression of x and y
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(x[2:], y[2:])
y_fit = [slope * xi + intercept for xi in x]
plt.plot(x, y_fit, color='red', label='Linear fit, k={:.2f}, r^2={:.2f}'.format(slope, r_value**2))
plt.legend()
plt.savefig(f'Picture/Log_inFide_Vs_LogNm.svg')

plt.figure()
x = [(i) for i in Nm_list]
y = in_fide_ls
plt.ylim(0, max(y)*1.1)
plt.scatter(x[1:], y[1:])
plt.xlabel('N_m')
plt.ylabel('F_in')
plt.savefig(f'Picture/inFide_Vs_Nm.svg')
import numpy as np
import os
import matplotlib.pyplot as plt
import json

para = dict()
para['epoch'] = 200
para['num_f'] = 10
para['chi'] = 16
para['measure_train'] = 100
para['sample_num'] = 1000
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

Ns_list = [100 + 100*i for i in range(10)]
print(Ns_list)
Nm = 100

loss_ls = []
loss_log_ls = []
in_fide_ls = []
fide_ls = []

for Ns in Ns_list:
    path = os.path.join(result_path, f'{para["measure_train"]}_{Ns}_chi{para["chi"]}_miu{230}.json')
    loss, loss_log, fide = extract_from_json(path)
    loss_ls.append(loss)
    loss_log_ls.append(loss_log)
    in_fide_ls.append(1 - fide)
    fide_ls.append(fide)

plt.figure()
x = [(i) for i in Ns_list]
y = fide_ls
plt.ylim(0.95, 1)
plt.scatter(x, y)
plt.xlabel('N_s')
plt.ylabel('F')
plt.savefig(f'Picture/Fide_Vs_Ns.svg')

plt.figure()
x = [1/np.sqrt(i) for i in Ns_list]
y = in_fide_ls
plt.ylim(0, max(y)*1.1)
plt.scatter(x, y)
plt.xlabel('(N_s)^(-1/2)')
plt.ylabel('F_in')
plt.savefig(f'Picture/inFide_Vs_logNs.svg')

plt.figure()
x = [(i) for i in Ns_list]
y = in_fide_ls
plt.ylim(0, max(y)*1.1)
plt.scatter(x, y)
plt.xlabel('N_s')
plt.ylabel('F_in')
plt.savefig(f'Picture/inFide_Vs_Ns.svg')
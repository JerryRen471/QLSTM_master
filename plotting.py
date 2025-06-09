import torch as tc
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import copy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Nm', type=int, default=100)
parser.add_argument('--Ns', type=int, default=10000)
args = parser.parse_args()

para = dict()
para['epoch'] = 500
para['num_f'] = 10
para['chi'] = 16
para['measure_train'] = args.Nm
para['sample_num'] = args.Ns
para['hx'] = 1
chain_dir = f"chain{para['num_f']}"
result_path = os.path.join('Result', 'Rand_large', chain_dir, 'normal_1_3_0')
os.makedirs(f'Picture/{para["measure_train"]}_{para["sample_num"]}_chi{para["chi"]}_miu{230}', exist_ok=True)

# f = open('D:/lwj/code/python/QLSTM/Data2/Ising/chain%d/hx%g/%d_%d.txt' % (
#     para['num_f'], para['hx'], para['measure_train'], para['sample_num']), 'rb')
with open(os.path.join(result_path, f'{para["measure_train"]}_{para["sample_num"]}_chi{para["chi"]}_miu{230}.json'), 'rb') as f:
# f = open(os.path.join(result_path, f'{para["measure_train"]}_{para["sample_num"]}_chi{para["chi"]}_miu{230}.txt'), 'rb')
    dic = json.load(f)
# f.close()

def str2float(list_of_str):
    """
    Convert a list of strings to a list of floats.

    Args:
        list_of_str (list of str): List containing string representations of numbers.

    Returns:
        list of float: List containing the float representations.
    """
    return [float(s) for s in list_of_str]

trainloss_List = str2float(dic['trainloss_List'])
trainloss_log = str2float(dic['trainloss_log'])
fide_yang = str2float(dic['fide1'])
in_fide = [1 - fide for fide in fide_yang]

# testloss_List = dic['testloss_List']
# testloss_log = dic['testloss_log']
# ent0 = dic['ent0']
# ent1 = dic['ent1']
# ent2 = dic['ent2']
# efidelity = dic['efidelity']
# fide = dic['fide']
# cfidelity = dic['cfidelity']

'''画图'''
x1 = tc.linspace(0, para['epoch'] - 1, para['epoch'])
x2 = tc.linspace(1, para['epoch'] - 1, int(para['epoch']/2))
x3 = np.arange(len(fide_yang))

# z1 = testloss_List
# z2 = testloss_log

loss_min = min(trainloss_List)
loss_max = max(trainloss_List)
loss_log_min = min(trainloss_log)
loss_log_max = max(trainloss_log)
fide_min = min(fide_yang)
fide_max = max(fide_yang)

# b1 = min(z1)
# c1 = min(a1,b1)
# e1 = max(z1)
# f1 = max(d1,e1)
# b2 = min(z2)
# c2 = min(a2,b2)
# e2 = max(z2)
# f2 = max(d2,e2)


plt.figure()
plt.ylim(loss_min*0.9, loss_max+0.02*loss_max)
plt.scatter(x1.numpy(), trainloss_List)
# plt.scatter(x2.numpy(), z1)
plt.xlabel('Epoch')
plt.ylabel('MSELoss')
plt.title('Mps,L:%d,chi:%d'%(para['num_f'],para['chi']))
plt.savefig(f'Picture/{para["measure_train"]}_{para["sample_num"]}_chi{para["chi"]}_miu{230}/MSE_Loss.svg')

plt.figure()
plt.ylim(loss_log_min+0.01*loss_log_min, loss_log_max+0.01*loss_log_max)
plt.scatter(x1.numpy(), trainloss_log)
# plt.scatter(x2.numpy(), z2)
plt.xlabel('Epoch')
plt.ylabel('MSELoss_log')
plt.savefig(f'Picture/{para["measure_train"]}_{para["sample_num"]}_chi{para["chi"]}_miu{230}/MES_Loss_log.svg')

plt.figure()
plt.ylim(fide_min - 0.01*fide_min, 1)
plt.scatter(x3, fide_yang)
plt.xlabel('Epoch')
plt.ylabel('F')
plt.savefig(f'Picture/{para["measure_train"]}_{para["sample_num"]}_chi{para["chi"]}_miu{230}/Fide_yang.svg')

plt.figure()
plt.ylim(0, max(in_fide) * 1.1)
plt.scatter(x3, in_fide)
plt.xlabel('Epoch')
plt.ylabel('F_in')
plt.savefig(f'Picture/{para["measure_train"]}_{para["sample_num"]}_chi{para["chi"]}_miu{230}/in_Fide.svg')



# # "=============================="
# x1 = tc.linspace(0, para['epoch'] - 1, para['epoch'])
# x2 = tc.linspace(1, para['epoch'] - 1, int(para['epoch']/2))
# y1 = trainloss_nom
# z1 = testloss_nom
# y2 = trainloss_logn
# z2 = testloss_logn
# a1 = min(y1)
# b1 = min(z1)
# c1 = min(a1,b1)
# d1 = max(y1)
# e1 = max(z1)
# f1 = max(d1,e1)
# a2 = min(y2)
# b2 = min(z2)
# c2 = min(a2,b2)
# d2 = max(y2)
# e2 = max(z2)
# f2 = max(d2,e2)
#
# plt.figure()
# plt.ylim(0.95*c1, f1+0.02*f1)
# plt.scatter(x1.numpy(), y1)
# plt.scatter(x2.numpy(), z1)
# plt.xlabel('Epoch')
# plt.ylabel('MSELoss_nomalization')
# plt.title('Mps,L:%d,chi:%d'%(para['num_f'],para['chi']))
# plt.show()
# plt.figure()
# plt.ylim(c2+0.003*c2, 0.995*f2)
# plt.scatter(x1.numpy(), y2)
# plt.scatter(x2.numpy(), z2)
# plt.xlabel('Epoch')
# plt.ylabel('MSELoss_log_nom')
# plt.show()

# # "=============================="
# x = range(1, para['num_f'])
# plt.plot(x, ent0[1:para['num_f']], 'bs-.', markersize=6, linewidth=2, label='target')
# plt.plot(x, ent1[1:para['num_f']], 'rD:', markersize=7, linewidth=2, label='initial')
# plt.plot(x, ent2[1:para['num_f']], 'g*--', markersize=10, linewidth=2, label='final')
# plt.xlabel('Site')
# plt.ylabel('Entropy')
# plt.legend()
# plt.show()

# # "=============================="
# x5 = tc.linspace(1, para['epoch'] - 1, int(para['epoch']/2))
# y5 = tc.zeros([len(efidelity)], dtype=tc.float64)
# for i in range(len(efidelity)):
#     y5[i] = efidelity[i]
# e1 = min(y5)
# e2 = max(y5)
# plt.figure()
# plt.ylim(-0.05, e2+0.03*e2)
# plt.scatter(x5, y5)
# plt.xlabel('Epoch')
# plt.ylabel('Entropy_norm')
# plt.title('Mps,L:%d,chi:%d' % (para['num_f'], para['chi']))
# plt.show()

# # "=============================="
# x6 = tc.linspace(1, para['epoch'] - 1, int(para['epoch']/2))
# y6 = tc.zeros([len(fide)], dtype=tc.float64)
# for i in range(len(fide)):
#     y6[i] = fide[i]
# e1 = min(y6)
# e2 = max(y6)
# plt.figure()
# plt.ylim(-0.01, e2+0.03*e2)
# plt.scatter(x6, y6)
# plt.xlabel('Epoch')
# plt.ylabel('Fidelity')
# plt.title('Mps,L:%d,chi:%d' % (para['num_f'], para['chi']))
# plt.show()

# # "=============================="
# x7 = tc.linspace(1, para['epoch'] - 1, int(para['epoch']/2))
# y7 = tc.zeros([len(norm)], dtype=tc.float64)
# for i in range(len(norm)):
#     y7[i] = norm[i]
# m1 = min(y7)
# m2 = max(y7)
# plt.figure()
# plt.ylim(0.75, m2+0.03*m2)
# plt.scatter(x7, y7)
# plt.xlabel('Epoch')
# plt.ylabel('Norm')
# plt.title('Mps,L:%d,chi:%d' % (para['num_f'], para['chi']))
# plt.show()

# # "=============================="
# x8 = tc.linspace(1, para['epoch'] - 1, int(para['epoch']/2))
# y8 = tc.zeros([len(cfidelity)], dtype=tc.float64)
# for i in range(len(cfidelity)):
#     y8[i] = float(cfidelity[i])
# w1 = min(y8)
# w2 = max(y8)
# plt.figure()
# plt.ylim(w1-0.0001*w1, w2+0.04*w2)
# plt.scatter(x8, y8+0.03)
# plt.xlabel('Epoch')
# plt.ylabel('Classical Fidelity')
# plt.title('Mps,L:%d,chi:%d' % (para['num_f'], para['chi']))
# plt.show()



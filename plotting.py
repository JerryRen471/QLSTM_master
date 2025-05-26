import torch as tc
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

para = dict()
para['epoch'] = 600
para['num_f'] = 10
para['chi'] = 16
para['measure_train'] = 20
para['sample_num'] = 8192
para['hx'] = 1

# f = open('D:/lwj/code/python/QLSTM/Data2/Ising/chain%d/hx%g/%d_%d.txt' % (
#     para['num_f'], para['hx'], para['measure_train'], para['sample_num']), 'rb')
f = open('D:/lwj/code/python/QLSTM/Data2/Rand_Comp_Reluchi/chain%d/16/%d.txt' % (para['num_f'], para['measure_train']), 'rb')
dic = pickle.load(f)
f.close()

trainloss_List = dic['trainloss_List']
testloss_List = dic['testloss_List']
trainloss_log = dic['trainloss_log']
testloss_log = dic['testloss_log']
# ent0 = dic['ent0']
# ent1 = dic['ent1']
# ent2 = dic['ent2']
# efidelity = dic['efidelity']
# fide = dic['fide']
# cfidelity = dic['cfidelity']

'''画图'''
x1 = tc.linspace(0, para['epoch'] - 1, para['epoch'])
x2 = tc.linspace(1, para['epoch'] - 1, int(para['epoch']/2))
y1 = trainloss_List
z1 = testloss_List
y2 = trainloss_log
z2 = testloss_log
a1 = min(y1)
b1 = min(z1)
c1 = min(a1,b1)
d1 = max(y1)
e1 = max(z1)
f1 = max(d1,e1)
a2 = min(y2)
b2 = min(z2)
c2 = min(a2,b2)
d2 = max(y2)
e2 = max(z2)
f2 = max(d2,e2)

plt.figure()
plt.ylim(-0.0003, f1+0.02*f1)
plt.scatter(x1.numpy(), y1)
plt.scatter(x2.numpy(), z1)
plt.xlabel('Epoch')
plt.ylabel('MSELoss')
plt.title('Mps,L:%d,chi:%d'%(para['num_f'],para['chi']))
plt.show()
plt.figure()
plt.ylim(c2+0.01*c2, 0.975*f2)
plt.scatter(x1.numpy(), y2)
plt.scatter(x2.numpy(), z2)
plt.xlabel('Epoch')
plt.ylabel('MSELoss_log')
plt.show()

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



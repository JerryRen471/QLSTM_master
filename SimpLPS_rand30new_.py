import torch as tc
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import math
import copy

'''华为服务器运行'''
def Orthogonalize_left2right(mps, chi):
    type = mps[0].dtype
    device = mps[0].device
    for i in range(0, len(mps) - 1):
        merged_tensor = tc.tensordot(mps[i], mps[i + 1], ([3], [1]))
        s1 = merged_tensor.shape
        merged_tensor = merged_tensor.reshape(s1[0] * s1[1] * s1[2], s1[3] * s1[4] * s1[5])
        merged_tensor = merged_tensor.cpu().numpy()
        u, lm, v = np.linalg.svd(merged_tensor, full_matrices=False)
        u, lm, v = tc.from_numpy(u), tc.from_numpy(lm), tc.from_numpy(v)
        u, lm, v = u.to(dtype=type, device=device), lm.to(dtype=type, device=device), v.to(dtype=type, device=device)
        bdm = min(chi, len(lm))
        u = u[:, :bdm]
        lm = tc.diag(lm[:bdm])
        v = v[:bdm, :]
        mps[i] = u.reshape(s1[0], s1[1], s1[2], bdm)
        v = tc.mm(lm, v)
        s2 = mps[i + 1].shape
        mps[i + 1] = v.reshape(v.shape[0], s2[0], s2[2], s2[3])
        mps[i + 1] = tc.transpose(mps[i + 1], 0, 1)
    return mps


def Orthogonalize_right2left(mps, chi):
    type = mps[0].dtype
    device = mps[0].device
    for i in range(len(mps) - 1, 0, -1):
        merged_tensor = tc.tensordot(mps[i - 1], mps[i], ([3], [1]))
        s1 = merged_tensor.shape
        merged_tensor = merged_tensor.reshape(s1[0] * s1[1] * s1[2], s1[3] * s1[4] * s1[5])
        merged_tensor = merged_tensor.cpu().numpy()
        u, lm, v = np.linalg.svd(merged_tensor, full_matrices=False)
        u, lm, v = tc.from_numpy(u), tc.from_numpy(lm), tc.from_numpy(v)
        u, lm, v = u.to(dtype=type, device=device), lm.to(dtype=type, device=device), v.to(dtype=type, device=device)
        bdm = min(chi, len(lm))
        u = u[:, :bdm]
        lm = tc.diag(lm[:bdm])
        v = v[:bdm, :]
        mps[i] = v.reshape(v.shape[0], s1[3], s1[4], s1[5])
        mps[i] = tc.transpose(mps[i], 0, 1)
        u = tc.mm(u, lm)
        s = mps[i - 1].shape
        mps[i - 1] = u.reshape(s[0], s[1], s[2], u.shape[1])
    return mps


def calculate_inner_product_new(mps_l0, mps_l1):
    MPS_l0 = copy.deepcopy(mps_l0)
    s0 = MPS_l0[0].shape
    MPS_l0[0] = MPS_l0[0].reshape(s0[0], s0[2], s0[3])
    MPS_l1 = copy.deepcopy(mps_l1)
    s1 = MPS_l1[0].shape
    MPS_l1[0] = MPS_l1[0].reshape(s1[0], s1[2], s1[3])
    tmp0 = MPS_l0[0]
    tmp0 = tc.einsum('abc,ade->bcde', [tmp0, MPS_l0[0].conj()])
    tmp0 = tc.einsum('bcde,fdg->bcefg', [tmp0, MPS_l1[0]])
    tmp0 = tc.einsum('bcefg,fbh->cegh', [tmp0, MPS_l1[0].conj()])
    for i in range(1, len(MPS_l0)):
        tmp0 = tc.einsum('abcd,eafg->bcdefg', [tmp0, MPS_l0[i]])
        tmp0 = tc.einsum('bcdefg,ebhk->cdfghk', [tmp0, MPS_l0[i].conj()])
        tmp0 = tc.einsum('cdfghk,ochm->dfgkom', [tmp0, MPS_l1[i]])
        tmp0 = tc.einsum('dfgkom,odfq->gkmq', [tmp0, MPS_l1[i].conj()])
    inner_product = tc.squeeze(tmp0)
    return inner_product


def calculate_fidelity_yang(mps_l0, mps_l1):
    MPS_l0 = copy.deepcopy(mps_l0)
    MPS_l1 = copy.deepcopy(mps_l1)
    rho12 = calculate_inner_product_new(MPS_l0, MPS_l1)
    rho11 = calculate_inner_product_new(MPS_l0, MPS_l0)
    rho22 = calculate_inner_product_new(MPS_l1, MPS_l1)
    f = rho12 / (rho11 * rho22) ** 0.5
    return f


'''加载随机目标态'''
f = open('/Users/wenjun/code/Data/target_state/Rand_large/chain10/state10_normal_1_3_0.pr', 'rb')
rhostate = pickle.load(f)
f.close()
for i in range(len(rhostate)):
    rhostate[i] = tc.from_numpy(rhostate[i]).cpu()


def SimpleMPS(train_dataset, para=None):
    para_def = dict()
    para_def['num_f'] = 30
    para_def['chi'] = 16
    para_def['d'] = 2
    para_def['miu1'] = 1
    para_def['miu2'] = 2
    para_def['seed'] = 0

    para_def['epoch'] = 500
    para_def['batch'] = 5000
    para_def['lr'] = 3.0 * 1e-3
    para['if_fidelity_yang'] = True
    para_def['fidelity_yang_epoch'] = 2

    para_def['device'] = 'cpu'  # 'cuda'
    para_def['dtype'] = tc.complex128
    para = dict(para_def, **para)

    '''数据按照batch切割'''
    train_loader = DataLoader(train_dataset, batch_size=para['batch'], shuffle=True)

    '''定义损失函数和优化器'''
    classifier = sMPS(para['num_f'], para['chi'], para['d'], para['miu1'], para['miu2'], para['seed'],
                      device=para['device'], dtype=para['dtype'])
    optimizer = tc.optim.Adam(classifier.mps_l, lr=para['lr'])
    loss_func = nn.MSELoss()  # nn.CrossEntropyLoss()

    '''训练'''
    trainloss_List = []
    trainloss_log = []
    fide1 = []

    for t in range(para['epoch']):  # epoch
        print('\n-------- Epoch: %d --------' % t)
        train_loss = 0
        train_tot = 0
        for batch_idx, (samples, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            # samples格式应为 (num_samples, num_features, 2)
            outputs = classifier.prob(samples)
            loss = loss_func(outputs, targets)
            train_loss += loss.data.item() * samples.shape[0]
            train_tot += samples.shape[0]
            loss.backward()
            optimizer.step()
        print(train_loss / train_tot)
        print(np.log(train_loss / train_tot))
        trainloss_List.append(train_loss / train_tot)
        trainloss_log.append(np.log(train_loss / train_tot))

        '''计算Fidelity_yang'''
        if para['if_fidelity_yang'] and ((t + 1) % para['fidelity_yang_epoch'] == 0):
            classifier.eval()
            with tc.no_grad():
                mps_l6 = copy.deepcopy(classifier.mps_l)
                Orthogonalize_left2right(mps_l6, chi=para['chi'])
                Orthogonalize_right2left(mps_l6, chi=para['chi'])
                fide_new = calculate_fidelity_yang(mps_l6, rhostate)
                print('fidelity_yang')
                print('%.16f' % abs(fide_new).cpu())
                fide1.append('%.16f' % abs(fide_new).cpu())
            classifier.train()

    dic = {'trainloss_List': trainloss_List, 'trainloss_log': trainloss_log, 'fyang': fide1}
    f = open('/Users/wenjun/code/Result/Rand_large/chain%d/normal_1_3_0/%d_%d_chi%d.txt' % (
        para['num_f'], para['measure_train'], para['sample_num'], para['chi']), 'wb')
    pickle.dump(dic, f)
    f.close()

    return {'trainloss_List': trainloss_List, 'trainloss_log': trainloss_log, 'fyang': fide1}


class sMPS(nn.Module):

    def __init__(self, length, chi, d, miu1, miu2, seed, device=None, dtype=None):
        super(sMPS, self).__init__()
        self.l = length
        self.chi = chi
        self.d = d
        self.miu1 = miu1
        self.miu2 = miu2
        self.device = device
        self.dtype = dtype
        tc.manual_seed(seed)
        self.mps_l = list(range(self.l))
        self.mps_l[0] = tc.randn((self.miu2, 1, self.d, self.chi), device=self.device, dtype=self.dtype)
        self.mps_l[-1] = tc.randn((self.miu2, self.chi, self.d, 1), device=self.device, dtype=self.dtype)
        for n in range(1, self.l - 1):
            self.mps_l[n] = tc.randn((self.miu2, self.chi, self.d, self.chi), device=self.device, dtype=self.dtype)
        Orthogonalize_left2right(self.mps_l, chi=self.chi)
        Orthogonalize_right2left(self.mps_l, chi=self.chi)
        self.mps_l[0] = self.mps_l[0] / tc.norm(self.mps_l[0])
        for n in range(self.l):
            print(self.mps_l[n].shape)
            self.mps_l[n].requires_grad = True

    def prob(self, x):
        tmp_l = list(range(self.l))
        for i in range(0, self.l):
            tmp_l[i] = tc.einsum('abcd,nc->nabd', [self.mps_l[i], x[:, i, :].conj()])
        tmp_e = tc.einsum('nabc,nade->nbcde', [tmp_l[0], tmp_l[0].conj()])
        for i in range(1, self.l):
            tmp = tc.einsum('nabcd,nebf->nafcde', [tmp_e, tmp_l[i]])
            tmp_e = tc.einsum('nafcde,nedg->nafcg', [tmp, tmp_l[i].conj()])
        p = tc.abs(tmp_e[:, 0].reshape(tmp_e[:, 0].shape[0], tmp_e[:, 0].shape[1])).to(tc.float64)
        return p


def run():
    """
    ===================================
    以下为调用样例,分为载入数据和调用程序两步
    ===================================
    """
    para = dict()
    para['num_f'] = 10  # lps_len
    para['chi'] = 16
    para['d'] = 2
    para['miu1'] = 1
    para['miu2'] = 2
    para['seed'] = 0

    para['epoch'] = 200
    para['batch'] = 10000  # iteration=30-40
    para['lr'] = 1 * 1e-3

    para['measure_train'] = 20
    para['sample_num'] = 1000
    para['device'] = 'cpu'  # 'cuda'
    para['dtype'] = tc.complex128

    '''加载数据集'''
    xy = np.loadtxt('/Users/wenjun/code/Data/Random_Glps/chain%d/normal_1_3_0/Nm/%d_%d.txt' % (
        para['num_f'], para['measure_train'], para['sample_num']), delimiter=',', dtype=np.float64, skiprows=1)
    f = open('/Users/wenjun/code/Data/Random_Glps/chain%d/normal_1_3_0/Nm/%d_%d.txt' % (
        para['num_f'], para['measure_train'], para['sample_num']))
    train_test_num = list(range(1))  # 读取第一行
    for i in range(1):
        line = f.readline().strip()
        data_list = []
        num = list(map(float, line.split()))
        data_list.append(num)
        train_test_num[i] = int(np.array(data_list))
    train_num = train_test_num[0]
    print('Size of the  dataset is %s' % train_num)
    x_train = tc.from_numpy(xy[0:train_num, 0:-1])
    y_train = tc.from_numpy(xy[0:train_num, [-1]]).to(para['device'])
    # Feature Map
    a = 1 / math.sqrt(2)
    b = tc.tensor(1j / math.sqrt(2), dtype=tc.complex128)
    # 创建一个映射表
    mapping_table = tc.tensor([
        [a, a],  # 0
        [a, -a],  # 1
        [a, b],  # 2
        [a, -b],  # 3
        [1, 0],  # 4
        [0, 1],  # 5
        ], device=para['device'], dtype=para['dtype'])
    data_mapping = tc.tensor(x_train, dtype=tc.long)
    data_mapped = mapping_table[data_mapping]  # 直接索引映射表，得到结果
    train_dataset = TensorDataset(data_mapped, y_train)

    Rst = SimpleMPS(train_dataset, para)


if __name__ == '__main__':
    run()
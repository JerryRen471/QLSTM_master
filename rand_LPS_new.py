import numpy as np
import torch as tc
import pickle
import copy
import os

'''LPS正则化'''
def Orthogonalize_left2right(lps_l, chi):
    type = lps_l[0].dtype
    for i in range(0, len(lps_l) - 1):
        merged_tensor = tc.tensordot(lps_l[i], lps_l[i+1], ([3], [1]))
        s1 = merged_tensor.shape
        merged_tensor = merged_tensor.reshape(s1[0]*s1[1]*s1[2], s1[3]*s1[4]*s1[5])
        merged_tensor = merged_tensor.cpu().numpy()
        u, lm, v = np.linalg.svd(merged_tensor,full_matrices=False)
        u,lm,v = tc.from_numpy(u),tc.from_numpy(lm),tc.from_numpy(v)
        u, lm, v = u.to(dtype=type,device=device), lm.to(dtype=type,device=device), v.to(dtype=type,device=device)
        bdm = min(chi, len(lm))
        u = u[:, :bdm]
        lm = tc.diag(lm[:bdm])
        v = v[:bdm, :]
        lps_l[i] = u.reshape(s1[0], s1[1], s1[2], bdm)
        v = tc.mm(lm, v)
        s2 = lps_l[i + 1].shape
        lps_l[i+1] = v.reshape(v.shape[0], s2[0], s2[2], s2[3])
        lps_l[i + 1] = tc.transpose(lps_l[i+1],0,1)
    return lps_l

def Orthogonalize_right2left(lps_l, chi):
    type = lps_l[0].dtype
    for i in range(len(lps_l)-1, 0, -1):
        merged_tensor = tc.tensordot(lps_l[i-1], lps_l[i], ([3], [1]))
        s1 = merged_tensor.shape
        merged_tensor = merged_tensor.reshape(s1[0]*s1[1]*s1[2], s1[3]*s1[4]*s1[5])
        merged_tensor = merged_tensor.cpu().numpy()
        u, lm, v = np.linalg.svd(merged_tensor,full_matrices=False)
        u, lm, v = tc.from_numpy(u), tc.from_numpy(lm), tc.from_numpy(v)
        u, lm, v = u.to(dtype=type, device=device), lm.to(dtype=type, device=device), v.to(dtype=type, device=device)
        bdm = min(chi, len(lm))
        u = u[:, :bdm]
        lm = tc.diag(lm[:bdm])
        v = v[:bdm, :]
        lps_l[i] = v.reshape(v.shape[0], s1[3], s1[4], s1[5])
        lps_l[i] = tc.transpose(lps_l[i], 0, 1)
        u = tc.mm(u, lm)
        s = lps_l[i-1].shape
        lps_l[i-1] = u.reshape(s[0], s[1], s[2], u.shape[1])
    return lps_l

def calculate_inner_product_new(lps_l0,lps_l1):
    LPS_l0 = copy.deepcopy(lps_l0)
    s0 = LPS_l0[0].shape
    LPS_l0[0] = LPS_l0[0].reshape(s0[0], s0[2], s0[3])
    LPS_l1 = copy.deepcopy(lps_l1)
    s1 = LPS_l1[0].shape
    LPS_l1[0] = LPS_l1[0].reshape(s1[0], s1[2], s1[3])
    tmp0 = LPS_l0[0]
    tmp0 = tc.einsum('abc,ade->bcde', [tmp0,LPS_l0[0].conj()])
    tmp0 = tc.einsum('bcde,fdg->bcefg', [tmp0,LPS_l1[0]])
    tmp0 = tc.einsum('bcefg,fbh->cegh', [tmp0,LPS_l1[0].conj()])
    for i in range(1, len(LPS_l0)):
        tmp0 = tc.einsum('abcd,eafg->bcdefg', [tmp0,LPS_l0[i]])
        tmp0 = tc.einsum('bcdefg,ebhk->cdfghk', [tmp0, LPS_l0[i].conj()])
        tmp0 = tc.einsum('cdfghk,ochm->dfgkom', [tmp0, LPS_l1[i]])
        tmp0 = tc.einsum('dfgkom,odfq->gkmq', [tmp0, LPS_l1[i].conj()])
    inner_product = tc.squeeze(tmp0)
    return inner_product

l = 10
chi = 16
d = 2
miu = 2
means = 1
stds = 3
seeds = 0
device = 'cpu'
dtype = tc.complex128
tc.manual_seed(seeds)
lps_l = list(range(l))
lps_l[0] = tc.normal(mean=means,std=stds,size=(miu, 1, d, chi), device=device, dtype=dtype)
lps_l[-1] = tc.normal(mean=means,std=stds,size=(miu, chi, d, 1), device=device, dtype=dtype)
for n in range(1, l - 1):
    lps_l[n] = tc.normal(mean=means,std=stds,size=(miu, chi, d, chi), device=device, dtype=dtype)
Orthogonalize_left2right(lps_l, chi=chi)
Orthogonalize_right2left(lps_l, chi=chi)
lps_l[0] = lps_l[0] / tc.norm(lps_l[0])

inner = calculate_inner_product_new(lps_l,lps_l)
print(inner)

for i in range(l):
    lps_l[i] = lps_l[i].cpu().numpy()
    print(lps_l[i].shape)

# Use local directory for saving
dir_path = "Data/target_state/Rand_large/chain%d" % l

parent_dir = os.path.dirname(dir_path)
if not os.access(parent_dir, os.W_OK):
    print("当前进程无权限在父目录中创建子目录")
    # 执行相应的错误处理逻辑
    ...
else:
    try:
        os.makedirs(dir_path, exist_ok=True)
        print("目录创建成功")
    except PermissionError:
        print("当前进程无权限在目标目录中创建子目录")
        # 执行相应的错误处理逻辑
        ...

# Check if directory exists and create if it doesn't
save_dir = dir_path
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")

f = open(save_dir+'/state%d_normal_%d_%d_%d.pr'%(l, means, stds, seeds), 'wb')
pickle.dump(lps_l, f)
f.close()
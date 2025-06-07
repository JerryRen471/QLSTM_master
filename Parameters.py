import torch as tc
import os

def measure_data(para=dict()):
    para_def = dict()
    para_def['retained_feature'] = 10
    para_def['sample_num'] = 10000
    para_def['measure_mode_init'] = 'True'
    para_def['measure_num_init'] = 10
    para_def['measure_step'] = 10
    para_def['measure_num_tot'] = 100
    para_def['device'] = 'cuda:0'
    para_def['dtype'] = tc.complex64
    para = dict(para_def, **para)
    para['load_state_path'] = 'Data/target_state/Rand_large/chain%d/state%d_normal_1_3_0.pr' % (para['retained_feature'], para['retained_feature'])
    para['save_Nmdata_path'] = 'Data/Random_Glps/chain%d/normal_1_3_0/Nm/' % (para['retained_feature'])
    if not os.path.exists(para['save_Nmdata_path']):
        os.makedirs(para['save_Nmdata_path'])
    if not os.path.exists(para['load_state_path']):
        os.makedirs(para['load_state_path'])
    return para


def sample_data(para=dict()):
    para_def = dict()
    para_def.update(measure_data())
    para_def['measure_train'] = 100
    para_def['sample_num_init'] = 1000
    para_def['sample_step'] = 1000
    para_def['sample_num_tot'] = 10000
    para = dict(para_def, **para)
    para['sample_mode_init'] = 'True'
    para['save_Nsdata_path'] = 'Data/Random_Glps/chain%d/normal_1_3_0/Ns/' % (para_def['retained_feature'])
    if not os.path.exists(para['save_Nsdata_path']):
        os.makedirs(para['save_Nsdata_path'])
    return para
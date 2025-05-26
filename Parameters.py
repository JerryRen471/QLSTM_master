import torch as tc

def measure_data(para=dict()):
    para['retained_feature'] = 10
    para['sample_num'] = 100
    para['measure_mode_init'] = 'True'
    para['measure_num_init'] = 10
    para['measure_step'] = 5
    para['measure_num_tot'] = 20
    para['load_state_path'] = '/Data/target_state/Rand_large/chain%d/state%d_normal_1_3_0.pr' % (para['retained_feature'], para['retained_feature'])
    para['save_Nmdata_path'] = '/Data/Random_Glps/chain%d/normal_1_3_0/Nm/' % (para['retained_feature'])
    para['device'] = 'cpu'
    para['dtype'] = tc.complex64
    return para


def sample_data(para=dict()):
    para.update(measure_data())
    para['measure_train'] = 10
    para['sample_num_init'] = 100
    para['sample_step'] = 50
    para['sample_num_tot'] = 200
    para['sample_mode_init'] = 'True'
    para['save_Nsdata_path'] = '/Data/Random_Glps/chain%d/normal_1_3_0/Ns/' % (para['retained_feature'])
    return para
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pandas as pd

para = dict()
para['epoch'] = 200
para['num_f'] = 10
para['chi'] = 16
para['measure_train'] = 100
para['sample_num'] = 10000
para['hx'] = 1
chain_dir = f"chain{para['num_f']}"
result_path = os.path.join('Result', 'Rand_large', chain_dir, 'normal_1_3_0')

def extract_from_csv(path, para_columns, stats_columns):
    """
    Extract and process data from CSV file.
    
    Args:
        path (str): Path to the CSV file
        para_columns (list): List of column names to group by
        stats_columns (list): List of column names to calculate statistics for
        
    Returns:
        dict: Dictionary containing mean and variance for each group
    """
    # Read CSV file
    df = pd.read_csv(path)
    
    # Group by para_columns and calculate statistics
    grouped = df.groupby(para_columns)[stats_columns].agg(['mean', 'var'])
    
    # Convert to dictionary format
    result = {}
    for group in grouped.index:
        group_key = tuple(group) if len(para_columns) > 1 else group
        stats_dic = dict()
        for stat in stats_columns:
            stats_dic[stat] = {
                'mean': grouped.loc[group][stat]['mean'],
                'var': grouped.loc[group][stat]['var'],
                }
        result[group_key] = stats_dic
    return result

# Ns_list = [100 + 100*i for i in range(10)]
# print(Ns_list)
Ns = 10000

loss_ls = []
loss_log_ls = []
in_fide_ls = []
fide_ls = []

path = 'Result/Rand_large/chain10/normal_1_3_0/TrainResult.csv'
df = pd.read_csv(path)
fide_group = df.groupby('sample_num')['fide1_mean'].agg(['mean', 'var'])
print(fide_group['mean'])
Ns_list = fide_group['mean'].index.values
fide_mean = fide_group['mean'].values
fide_var = fide_group['var'].values

plt.figure()
x = [(i) for i in Ns_list]
y = fide_mean
yerr = np.sqrt(fide_var)  # 标准差作为误差棒
plt.ylim(min(y), 1)
plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, label='Fidelity with error bar')
plt.xlabel('N_s')
plt.ylabel('F')
plt.legend()
plt.savefig(f'Picture/Fide_Vs_Ns_with_errorbar.pdf')

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
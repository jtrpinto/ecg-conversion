'''
Script to process PTB-XL data (with medical condition annotations).


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''


import os
import ast
import wfdb
import numpy as np
import pandas as pd
import pickle as pk


SAVE = 'dataset_info.pk'        # pickle in which to save info
FOLD = [1,2,3,4,5,6,7,8,9,10]   # PTB-XL folds to use 1-10
SHARE = 1.0                     # share of data to use [0., 1.]

PTBXL_PATH = 'PTB-XL'           # path to PTB-XL folder


def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def select_data(Y, fold, share=1.0):
    np.random.seed(42)
    Y = Y[(Y.strat_fold == fold)]  # Selecting from fold
    Y = Y.loc[(len(a) == 1 for a in Y.diagnostic_superclass)]  # Discarding those that have multiple labels
    if share < 1.0:
       Y = Y.loc[(np.random.uniform() < share for a in Y.strat_fold)] 
    paths = list(Y.filename_hr)
    annts = list(Y.diagnostic_superclass)
    return paths, annts

def load_raw_data(file, path):
    data, _ = wfdb.rdsamp(path + file)
    return data


Y = pd.read_csv(os.path.join(PTBXL_PATH, 'ptbxl_database.csv'), index_col='ecg_id')

Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))


# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(os.path.join(PTBXL_PATH, 'scp_statements.csv'), index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

total_paths = list()
total_annts = list()

for fold in FOLD:
    paths, annts = select_data(Y, fold, share=SHARE)

    total_paths.append(paths)
    total_annts.append(annts)

paths = np.concatenate(total_paths)
annts = np.concatenate(total_annts)

print(len(paths))
print(np.unique(annts, return_counts=True))

with open(SAVE, 'wb') as hf:
    pk.dump((paths, annts), hf)

lead_i = list()
lead_ii = list()
lead_iii = list()
lead_avr = list()
lead_avl = list()
lead_avf = list()
lead_v1 = list()
lead_v2 = list()
lead_v3 = list()
lead_v4 = list()
lead_v5 = list()
lead_v6 = list() 

for ll in range(len(paths)):
    data = load_raw_data(paths[ll], PTBXL_PATH)
    lead_i.append(data[0:2500, 0])
    lead_ii.append(data[0:2500, 1])
    lead_iii.append(data[0:2500, 2])
    lead_avl.append(data[0:2500, 3])
    lead_avr.append(data[0:2500, 4])
    lead_avf.append(data[0:2500, 5])
    lead_v1.append(data[0:2500, 6])
    lead_v2.append(data[0:2500, 7])
    lead_v3.append(data[0:2500, 8])
    lead_v4.append(data[0:2500, 9])
    lead_v5.append(data[0:2500, 10])
    lead_v6.append(data[0:2500, 11]) 
    print('... recording', ll+1, '/', len(paths))

with open(os.path.join(PTBXL_PATH, 'lead_i.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_i), 'y_test': annts}, hf)

with open(os.path.join(PTBXL_PATH, 'lead_ii.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_ii), 'y_test': annts}, hf)

with open(os.path.join(PTBXL_PATH, 'lead_iii.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_iii), 'y_test': annts}, hf)

with open(os.path.join(PTBXL_PATH, 'lead_avr.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_avr), 'y_test': annts}, hf)

with open(os.path.join(PTBXL_PATH, 'lead_avl.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_avl), 'y_test': annts}, hf)

with open(os.path.join(PTBXL_PATH, 'lead_avf.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_avf), 'y_test': annts}, hf)

with open(os.path.join(PTBXL_PATH, 'lead_v1.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_v1), 'y_test': annts}, hf)

with open(os.path.join(PTBXL_PATH, 'lead_v2.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_v2), 'y_test': annts}, hf)

with open(os.path.join(PTBXL_PATH, 'lead_v3.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_v3), 'y_test': annts}, hf)

with open(os.path.join(PTBXL_PATH, 'lead_v4.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_v4), 'y_test': annts}, hf)

with open(os.path.join(PTBXL_PATH, 'lead_v5.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_v5), 'y_test': annts}, hf)

with open(os.path.join(PTBXL_PATH, 'lead_v6.pk'), 'wb') as hf:
    pk.dump({'X_test': np.array(lead_v6), 'y_test': annts}, hf)

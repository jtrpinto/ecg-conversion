'''
Script to compute average signal correlations between one reference lead and all the others.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.

'''


import sys
import torch
import numpy as np

from datasets import Dataset
from losses import corr


DB = 'PTB-XL'           # database to use
LEAD_IN = 'I'           # reference lead
FS = 500.0              # data sampling frequency


# Use GPU if one is available
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(DEVICE)


dataset = Dataset(FS, database=DB, lead_in=LEAD_IN, dataset='test')
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

leads_list = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

with torch.no_grad():
    # compute testing loss
    corrs = list()
    
    for i, (X_in, X_lead1, X_lead2, X_lead3, X_lead_avr, X_lead_avl, X_lead_avf, X_lead_v1, X_lead_v2, X_lead_v3, X_lead_v4, X_lead_v5, X_lead_v6, _) in enumerate(test_loader):
        X_in = X_in.float().to(DEVICE)
        X_lead1 = X_lead1.float().to(DEVICE)
        X_lead2 = X_lead2.float().to(DEVICE)
        X_lead3 = X_lead3.float().to(DEVICE)
        X_lead_avr = X_lead_avr.float().to(DEVICE)
        X_lead_avl = X_lead_avl.float().to(DEVICE)
        X_lead_avf = X_lead_avf.float().to(DEVICE)
        X_lead_v1 = X_lead_v1.float().to(DEVICE)
        X_lead_v2 = X_lead_v2.float().to(DEVICE)
        X_lead_v3 = X_lead_v3.float().to(DEVICE)
        X_lead_v4 = X_lead_v4.float().to(DEVICE)
        X_lead_v5 = X_lead_v5.float().to(DEVICE)
        X_lead_v6 = X_lead_v6.float().to(DEVICE)

        X_target = [X_lead1, X_lead2, X_lead3,
                    X_lead_avr, X_lead_avl, X_lead_avf,
                    X_lead_v1, X_lead_v2, X_lead_v3,
                    X_lead_v4, X_lead_v5, X_lead_v6]

        corrs.append([float(corr([X_in], [X_target[ll]], reduce=False)[0].detach().cpu().numpy()) for ll in range(len(X_target))])

        sys.stdout.write('\r' + '........test sample {} / {} '.format(i + 1, len(dataset)))
        sys.stdout.flush()

corrs = np.array(corrs)

print()
print('LEAD IN: ', LEAD_IN)
print('R (avg.)', np.mean(corrs, axis=0))
print()

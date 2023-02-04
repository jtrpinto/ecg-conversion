'''
Script to split PTB-XL test results by medical diagnosis superclass annotation.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''


import numpy as np
import pickle as pk


RESULTS = 'results_file.pk'     # pickle storing the PTB-XL results, created by test.py


with open(RESULTS, 'rb') as hf:
    res = pk.load(hf)[1]

with open('PTB-XL/lead_i.pk', 'rb') as hf:
    ann = pk.load(hf)['y_test']

norm_res = res[(ann=='NORM').reshape(-1), :]
hyp_res = res[(ann=='HYP').reshape(-1), :]
cd_res = res[(ann=='CD').reshape(-1), :]
sttc_res = res[(ann=='STTC').reshape(-1), :]
mi_res = res[(ann=='MI').reshape(-1), :]

print(norm_res.shape)
print(hyp_res.shape)
print(cd_res.shape)
print(sttc_res.shape)
print(mi_res.shape)

print('NORM', np.mean(norm_res, axis=0))
print('MI', np.mean(mi_res, axis=0))
print('STTC', np.mean(sttc_res, axis=0))
print('CD', np.mean(cd_res, axis=0))
print('HYP', np.mean(hyp_res, axis=0))
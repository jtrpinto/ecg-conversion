'''
Experiment on diagnosis with reconstructed signals:
utility functions.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''


import numpy as np


def standard_normalisation(signal):
    # Returns signal with zero mean and unit variance
    return (signal - np.mean(signal)) / np.std(signal)


def stratified_train_validation_split(y, n_valid_per_class=1):
    # Divides the train dataset, assigning random n samples per class to the validation set.
    # n_valid_per_class can be >=1 (number of samples) or 0<n<1 (fraction of total identity samples)
    train_indices = list()
    valid_indices = list()
    for idd in np.unique(y):
        idd_indices = np.argwhere(y == idd)[:, 0]
        if n_valid_per_class >= 1:
            val_indices = np.random.choice(idd_indices, n_valid_per_class, replace=False)
        else:
            val_indices = np.random.choice(idd_indices, int(n_valid_per_class*len(idd_indices)), replace=False)
        for ii in idd_indices:
            if ii in val_indices:
                valid_indices.append(ii)
            else:
                train_indices.append(ii)
    return train_indices, valid_indices
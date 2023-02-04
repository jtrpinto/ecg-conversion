'''
Experiment on diagnosis with reconstructed signals:
dataset class to load PTB-XL diagnosis labelled data.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''


import pickle as pk
from torch.utils.data import Dataset
from utils import standard_normalisation
from scipy.signal import resample


class Dataset(Dataset):

    def __init__(self, signals_pickle, signals='original', dataset='train'):
        self.signals_pickle = signals_pickle
        self.dataset = dataset
        self.signals = signals

        if signals == 'original':
            with open(signals_pickle, 'rb') as handle:
                data = pk.load(handle)
                self.X = data[0]
                self.y = data[2]
        elif signals == 'reconstructed':
            with open(signals_pickle, 'rb') as handle:
                data = pk.load(handle)
                self.X = data[1]
                self.y = data[2]

        split_point = int(0.8 * len(self.X))

        if dataset == 'train':
            self.X = self.X[: split_point]
            self.y = self.y[: split_point]
        elif dataset == 'test':
            self.X = self.X[split_point :]
            self.y = self.y[split_point :]

                
    def __getitem__(self, index):
        x = standard_normalisation(resample(self.X[index, :], 4096)).reshape((1, 4096)).astype(float)
        y = int(self.y[index][0])
        return (x, y)
    
    def __len__(self):
        return self.X.shape[0]
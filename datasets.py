'''
Implementation of the dataset class to load the signal data.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.

'''


import numpy as np
import pickle as pk

from scipy import signal
from torch.utils.data import Dataset
from copy import deepcopy


class Dataset(Dataset):

    def __init__(self, fs, database='PTB', lead_in='I', dataset='train'):
        self.database = database
        if database=='PTB':
            files_leads = ['PTBfiles/ptb_data_i.pk', 'PTBfiles/ptb_data_ii.pk', 'PTBfiles/ptb_data_iii.pk',
                           'PTBfiles/ptb_data_avr.pk', 'PTBfiles/ptb_data_avl.pk', 'PTBfiles/ptb_data_avf.pk',
                           'PTBfiles/ptb_data_v1.pk', 'PTBfiles/ptb_data_v2.pk', 'PTBfiles/ptb_data_v3.pk',
                           'PTBfiles/ptb_data_v4.pk', 'PTBfiles/ptb_data_v5.pk', 'PTBfiles/ptb_data_v6.pk']
        elif database=='INCART':
            files_leads = ['INCART/incart_data_lead1.pk', 'INCART/incart_data_lead2.pk', 'INCART/incart_data_lead3.pk',
                           'INCART/incart_data_avr.pk', 'INCART/incart_data_avl.pk', 'INCART/incart_data_avf.pk',
                           'INCART/incart_data_v1.pk', 'INCART/incart_data_v2.pk', 'INCART/incart_data_v3.pk',
                           'INCART/incart_data_v4.pk', 'INCART/incart_data_v5.pk', 'INCART/incart_data_v6.pk']
        elif database=='PTB-XL':
            files_leads = ['PTB-XL/lead_i.pk', 'PTB-XL/lead_ii.pk', 'PTB-XL/lead_iii.pk',
                           'PTB-XL/lead_avr.pk', 'PTB-XL/lead_avl.pk', 'PTB-XL/lead_avf.pk',
                           'PTB-XL/lead_v1.pk', 'PTB-XL/lead_v2.pk', 'PTB-XL/lead_v3.pk',
                           'PTB-XL/lead_v4.pk', 'PTB-XL/lead_v5.pk', 'PTB-XL/lead_v6.pk']
        num_leads = len(files_leads)
        data_leads = list()

        for file_lead_n in files_leads:
            with open(file_lead_n, 'rb') as handle:
                data_leads.append(pk.load(handle))

        self.lead_in = lead_in

        data = list()
        keys = list()
        if dataset == 'train' or dataset == 'validation':
            for lead in range(num_leads):
                data.append(data_leads[lead]['X_train'])
            if database=='PTB':
                keys.append(data_leads[0]['y_train'])
            else: 
                keys=[np.zeros(len(data_leads[0]['X_train']))]
        elif dataset == 'test':
            for lead in range(num_leads):
                data.append(data_leads[lead]['X_test'])
            if database=='PTB':
                keys.append(data_leads[0]['y_test'])
            elif database=='PTB-XL':
                keys.append(data_leads[lead]['y_test'])
            else:
                keys=[np.zeros(len(data_leads[0]['X_test']))]
            
        else:
            raise ValueError('Variable dataset must be \'train\', \'validation\', or \'test\'.')
        self.keys = keys
        self.fs = fs
        self.X_in, self.X_lead1, self.X_lead2, self.X_lead3, self.X_lead_avr, self.X_lead_avl, self.X_lead_avf, self.X_lead_v1, self.X_lead_v2, self.X_lead_v3, self.X_lead_v4, self.X_lead_v5, self.X_lead_v6 = self.__create_dataset__(data)

    
    def __normalisation_sqrt__(self, X):
        # Returns signals with amplitude normalised between -1 and 1
        # X_norm = np.array([2*(sig - np.min(sig))/(np.max(sig) - np.min(sig) + 1e-6) - 1 for sig in X])
        X_norm = np.array([2*(sig - np.min(sig))/(np.max(sig) - np.min(sig) + 1e-9) - 1 for sig in X]) #[-1,1]
        X_norm = np.sqrt(X_norm/2 + 0.5)*2 - 1  # sqrt([0,1])*2-1
        return X_norm
    
        
    def __bandpass_filter__(self, segment, fs, fc=[1, 40]):
        # Filters the signal with a butterworth bandpass filter with cutoff frequencies fc=[a, b]
        f0 = 2 * float(fc[0]) / float(fs)
        f1 = 2 * float(fc[1]) / float(fs)
        b, a = signal.butter(2, [f0, f1], btype='bandpass')
        return signal.filtfilt(b, a, segment)

    def __NaNremoval__(self,X):
      #removes samples with NaN values and corresponding samples in other leads
      if np.isnan(X).any():
        np.argwhere(np.isnan(X))#acabar


    def __create_dataset__(self, data):
        # Not yet prepared for num_leads != 3
        lead1 = list()  #[ sample 331 lead1| | |]
        lead2 = list()  #[ sample 331 lead2| | |]
        lead3 = list()  #[ sample 331 lead3| | |]
        lead_avr = list()
        lead_avl = list()
        lead_avf = list()
        lead_v1 = list()
        lead_v2 = list()
        lead_v3 = list()
        lead_v4 = list()
        lead_v5 = list()
        lead_v6 = list()
        
        for jj in range(len(data[0])):
            lead1.append(self.__bandpass_filter__(data[0][jj], self.fs, fc=[1, 40]))
            lead2.append(self.__bandpass_filter__(data[1][jj], self.fs, fc=[1, 40]))
            lead3.append(self.__bandpass_filter__(data[2][jj], self.fs, fc=[1, 40]))
            lead_avr.append(self.__bandpass_filter__(data[3][jj], self.fs, fc=[1, 40]))
            lead_avl.append(self.__bandpass_filter__(data[4][jj], self.fs, fc=[1, 40]))
            lead_avf.append(self.__bandpass_filter__(data[5][jj], self.fs, fc=[1, 40]))
            lead_v1.append(self.__bandpass_filter__(data[6][jj], self.fs, fc=[1, 40]))
            lead_v2.append(self.__bandpass_filter__(data[7][jj], self.fs, fc=[1, 40]))
            lead_v3.append(self.__bandpass_filter__(data[8][jj], self.fs, fc=[1, 40]))
            lead_v4.append(self.__bandpass_filter__(data[9][jj], self.fs, fc=[1, 40]))
            lead_v5.append(self.__bandpass_filter__(data[10][jj], self.fs, fc=[1, 40]))
            lead_v6.append(self.__bandpass_filter__(data[11][jj], self.fs, fc=[1, 40]))
        
        lead1 = self.__normalisation_sqrt__(np.array(lead1))
        lead2 = self.__normalisation_sqrt__(np.array(lead2))
        lead3 = self.__normalisation_sqrt__(np.array(lead3))
        lead_avr = self.__normalisation_sqrt__(np.array(lead_avr))
        lead_avl = self.__normalisation_sqrt__(np.array(lead_avl))
        lead_avf = self.__normalisation_sqrt__(np.array(lead_avf))
        lead_v1 = self.__normalisation_sqrt__(np.array(lead_v1))
        lead_v2 = self.__normalisation_sqrt__(np.array(lead_v2))
        lead_v3 = self.__normalisation_sqrt__(np.array(lead_v3))
        lead_v4 = self.__normalisation_sqrt__(np.array(lead_v4))
        lead_v5 = self.__normalisation_sqrt__(np.array(lead_v5))
        lead_v6 = self.__normalisation_sqrt__(np.array(lead_v6))

        if self.lead_in == 'I':
            x_in = deepcopy(lead1)
        elif self.lead_in == 'II':
            x_in = deepcopy(lead2)

        return x_in, lead1, lead2, lead3, lead_avr, lead_avl, lead_avf, lead_v1, lead_v2, lead_v3, lead_v4, lead_v5, lead_v6

                
    def __getitem__(self, index):
        if self.database == 'PTB-XL':
            x_in = signal.resample(self.X_in[index, :], 5000).reshape((1, 5000)).astype(float)
            x_lead1 = signal.resample(self.X_lead1[index, :], 5000).reshape((1, 5000)).astype(float)
            x_lead2 = signal.resample(self.X_lead2[index, :], 5000).reshape((1, 5000)).astype(float)
            x_lead3 = signal.resample(self.X_lead3[index, :], 5000).reshape((1, 5000)).astype(float)
            
            x_lead_avr = signal.resample(self.X_lead_avr[index, :], 5000).reshape((1, 5000)).astype(float)
            x_lead_avl = signal.resample(self.X_lead_avl[index, :], 5000).reshape((1, 5000)).astype(float)
            x_lead_avf = signal.resample(self.X_lead_avf[index, :], 5000).reshape((1, 5000)).astype(float)
            
            x_lead_v1 = signal.resample(self.X_lead_v1[index, :], 5000).reshape((1, 5000)).astype(float)
            x_lead_v2 = signal.resample(self.X_lead_v2[index, :], 5000).reshape((1, 5000)).astype(float)
            x_lead_v3 = signal.resample(self.X_lead_v3[index, :], 5000).reshape((1, 5000)).astype(float)
            x_lead_v4 = signal.resample(self.X_lead_v4[index, :], 5000).reshape((1, 5000)).astype(float)
            x_lead_v5 = signal.resample(self.X_lead_v5[index, :], 5000).reshape((1, 5000)).astype(float)
            x_lead_v6 = signal.resample(self.X_lead_v6[index, :], 5000).reshape((1, 5000)).astype(float)
        else:
            x_in = self.X_in[index, :].reshape((1, self.X_in.shape[1])).astype(float)
            x_lead1 = self.X_lead1[index, :].reshape((1, self.X_lead1.shape[1])).astype(float)
            x_lead2 = self.X_lead2[index, :].reshape((1, self.X_lead2.shape[1])).astype(float)
            x_lead3 = self.X_lead3[index, :].reshape((1, self.X_lead3.shape[1])).astype(float)
            
            x_lead_avr = self.X_lead_avr[index, :].reshape((1, self.X_lead_avr.shape[1])).astype(float)
            x_lead_avl = self.X_lead_avl[index, :].reshape((1, self.X_lead_avl.shape[1])).astype(float)
            x_lead_avf = self.X_lead_avf[index, :].reshape((1, self.X_lead_avf.shape[1])).astype(float)
            
            x_lead_v1 = self.X_lead_v1[index, :].reshape((1, self.X_lead_v1.shape[1])).astype(float)
            x_lead_v2 = self.X_lead_v2[index, :].reshape((1, self.X_lead_v2.shape[1])).astype(float)
            x_lead_v3 = self.X_lead_v3[index, :].reshape((1, self.X_lead_v3.shape[1])).astype(float)
            x_lead_v4 = self.X_lead_v4[index, :].reshape((1, self.X_lead_v4.shape[1])).astype(float)
            x_lead_v5 = self.X_lead_v5[index, :].reshape((1, self.X_lead_v5.shape[1])).astype(float)
            x_lead_v6 = self.X_lead_v6[index, :].reshape((1, self.X_lead_v6.shape[1])).astype(float)

        key = self.keys[0][index]
        if self.database == 'PTB-XL':
            if key == 'NORM':
                key = 0
            elif key == 'MI':
                key = 1
            elif key == 'STTC':
                key = 2
            elif key == 'CD':
                key = 3
            elif key == 'HYP': 
                key = 4
        return (x_in, x_lead1, x_lead2, x_lead3, x_lead_avr, x_lead_avl, x_lead_avf, x_lead_v1, x_lead_v2, x_lead_v3, x_lead_v4, x_lead_v5, x_lead_v6, key)
    
    def __len__(self):
        return self.X_in.shape[0]

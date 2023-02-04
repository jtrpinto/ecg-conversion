'''
Script to train the proposed model for interlead conversion.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''


import torch
import numpy as np

from torch import nn
from torch import optim
from datasets import Dataset
from models import UNet
from trainers import train_model


# Some options and parameters:
SAVE_MODEL = "path_to_model"        # where to save the model

DATABASE = 'PTB'                    # name of the database (PTB, INCART, or PTB-XL)
FS = 1000.0                         # data sampling frequency
DATASET = 'train'                   # name of the dataset to get (train or test)

SHARED_ENC = True                   # whether or not to use a shared encoder
LEAD_IN = 'I'                       # input lead name

N_EPOCHS = 500                      # number of training epochs
BATCH_SIZE = 16                     # number of samples to get from the dataset at each iteration
VALID_SPLIT = 0.1                   # fraction of training samples to be used for validation
PATIENCE = 50                       # for early stopping

LEARN_RATE = 1e-4                   # learning rate
REG = 0.0                           # l2-regularization factor


# Use GPU if one is available
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


dataset = Dataset(FS, database=DATABASE, lead_in=LEAD_IN, dataset=DATASET)

# create data indices for training and validation splits
dataset_size = len(dataset)  # number of samples in training + validation sets

indices = list(range(dataset_size))
split = int(np.floor(VALID_SPLIT * dataset_size))  # samples in valid. set
np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]


train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=1,
                                           sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=1,
                                           sampler=valid_sampler)

print('\n ======= TRAINING MODEL UNET ' + SAVE_MODEL + ' ======= \n')

model = UNet(shared_enc=SHARED_ENC, bilinear=True).to(DEVICE)

loss_fn = nn.L1Loss().to(DEVICE)
optimiser = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=REG)

_ = train_model(model, loss_fn, optimiser, train_loader, N_EPOCHS, valid_loader=valid_loader, save_file=SAVE_MODEL, device=DEVICE, patience=PATIENCE)

'''
Experiment on diagnosis with reconstructed signals:
script to train diagnosis classifier model.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''


import torch
from torch import nn
from torch import optim
from models import DiagnosisModel
from trainers import train_model
from datasets import Dataset
from utils import stratified_train_validation_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


DSET_SHARED_FILE = '../pickle_with_shared_encoder_reconstructed_signals.pk'
DSET_INDIV_FILE = '../pickle_with_individual_encoder_reconstructed_signals.pk'

N_CLASSES = 5              # Number of classes

SAVE_MODEL = "path_to_model"   # Where to save the model

N_EPOCHS = 100            # number of training epochs
BATCH_SIZE = 32           # number of samples to get from the dataset at each iteration
VALID_SPLIT = 0.2         # number of enrollment samples per subject to be used for validation
PATIENCE = 10             # for early stopping

DROPOUT = 0.5     
LEARN_RATE = 1e-5  
REG = 1e-3 


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Building datasets
train_set = Dataset(DSET_SHARED_FILE, signals='original', dataset='train')

# creating data indices for training and validation splits
train_indices, valid_indices = stratified_train_validation_split(train_set.y, n_valid_per_class=VALID_SPLIT)

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=1,
                                           sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=1,
                                           sampler=valid_sampler)


# TRAINING THE MODEL ==============================================================================

print('\n ======= TRAINING MODEL ' + SAVE_MODEL + ' ======= \n')

model = DiagnosisModel(N=N_CLASSES, dropout=DROPOUT).to(DEVICE)

class_weights = torch.FloatTensor([1./8188, 1./2283, 1./2166, 1./1527, 1./480]).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimiser = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=REG)

out = train_model(model, loss_fn, optimiser, train_loader, N_EPOCHS, DEVICE, patience=PATIENCE, valid_loader=valid_loader, filename=SAVE_MODEL)

# TESTING THE MODEL ==============================================================================

model.load_state_dict(torch.load(SAVE_MODEL + '.pth', map_location=DEVICE))
model = model.to(DEVICE)

original_test_set = Dataset(DSET_SHARED_FILE, signals='original', dataset='test')
shared_test_set = Dataset(DSET_SHARED_FILE, signals='reconstructed', dataset='test')
indiv_test_set = Dataset(DSET_INDIV_FILE, signals='reconstructed', dataset='test')

true_labels = list()
pred_labels = list()

for sett in [original_test_set, shared_test_set, indiv_test_set]:
    test_loader = torch.utils.data.DataLoader(sett, batch_size=1, shuffle=False, num_workers=1)

    true = list()
    pred = list()

    model.eval()
    with torch.no_grad():
        test_loss = 0.
        t_corrects = 0
        t_total = 0
        for i, (X, y) in enumerate(test_loader):
            # copy the mini-batch to GPU
            X = X.float().to(DEVICE)
            y = y.to(DEVICE)
        
            ypred = model(X)                 # forward pass
            test_loss += loss_fn(ypred, y)  # accumulate the loss of the mini-batch
            t_corrects += (torch.argmax(ypred, 1) == y).float().sum()
            t_total += y.shape[0]

            true.append(y.item())
            pred.append(torch.argmax(ypred, 1).item())

        test_loss /= i + 1
        t_acc = t_corrects / t_total
        print('....test loss: {:.4f} :: acc {:.4f}'.format(test_loss.item(), t_acc))
    
    true_labels = true
    pred_labels.append(pred)


print('Original')
print(balanced_accuracy_score(true_labels, pred_labels[0]))
print(confusion_matrix(true_labels, pred_labels[0], normalize='true'))

print('Shared Encoder')
print(balanced_accuracy_score(true_labels, pred_labels[1]))
print(confusion_matrix(true_labels, pred_labels[1], normalize='true'))

print('Individual Encoder')
print(balanced_accuracy_score(true_labels, pred_labels[2]))
print(confusion_matrix(true_labels, pred_labels[2], normalize='true'))
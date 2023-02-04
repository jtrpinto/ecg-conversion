'''
Experiment on diagnosis with reconstructed signals:
script to evaluate diagnosis classifier model..


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''


import torch
from torch import nn
from models import DiagnosisModel
from datasets import Dataset
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


DSET_SHARED_FILE = '../pickle_with_shared_encoder_reconstructed_signals.pk'
DSET_INDIV_FILE = '../pickle_with_individual_encoder_reconstructed_signals.pk'

N_CLASSES = 5                   # Number of classes

SAVE_MODEL = "path_to_model"    # Where the model is saved

DROPOUT = 0.5    


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


model = DiagnosisModel(N=N_CLASSES, dropout=DROPOUT).to(DEVICE)

model.load_state_dict(torch.load(SAVE_MODEL + '.pth', map_location=DEVICE))
model = model.to(DEVICE)

class_weights = torch.FloatTensor([1./8188, 1./2283, 1./2166, 1./1527, 1./480]).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)


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
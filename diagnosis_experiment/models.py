'''
Experiment on diagnosis with reconstructed signals:
implementation of the simple classifier for diagnosis.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''


from torch import nn
import torch.nn.functional as F


class DiagnosisModel(nn.Module):
    def __init__(self, N=5, dropout=.0):
        super(DiagnosisModel, self).__init__()
        fd = 256
        self.convnet = nn.Sequential(nn.Conv1d(1, 64, 15, stride=1, padding=7),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(),
                                     nn.MaxPool1d(4),
                                     nn.Conv1d(64, 96, 15, stride=1, padding=7),
                                     nn.BatchNorm1d(96),
                                     nn.ReLU(),
                                     nn.MaxPool1d(4),
                                     nn.Conv1d(96, 128, 15, stride=1, padding=7),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.MaxPool1d(4),
                                     nn.Conv1d(128, 160, 15, stride=1, padding=7),
                                     nn.BatchNorm1d(160),
                                     nn.ReLU(),
                                     nn.MaxPool1d(4),
                                     nn.Conv1d(160, 192, 15, stride=1, padding=7),
                                     nn.BatchNorm1d(192),
                                     nn.ReLU(),
                                     nn.MaxPool1d(4),
                                     nn.Conv1d(192, 256, 15, stride=1, padding=7),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(),
                                     nn.MaxPool1d(4)
                                    )

        self.fc = nn.Sequential(nn.Linear(fd, 1024),
                                nn.ReLU(),
                                nn.Dropout(p=dropout),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Dropout(p=dropout),
                                nn.Linear(512, N)
                               )

    def forward(self, x):
        h = self.convnet(x)
        h = h.view(h.size()[0], -1)
        output = self.fc(h)
        return output
    
    def predict(self, X):
        logits = self.forward(X)
        probs = F.softmax(logits, dim=1)
        return probs
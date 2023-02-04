'''
Implementation of the baseline model [1] for interlead conversion.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.


----
[1] Grande-Fidalgo A, Calpe J, Redón M, Millán-Navarro C, Soria-Olivas E.
Lead reconstruction using artifcial neural networks for ambulatory ECG
acquisition. Sensors. 2021. https://doi.org/10.3390/s21165542

'''


from torch import nn


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        
        self.ann = nn.Sequential(nn.Linear(1, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 12),
                                 nn.Tanh()
                                )

    def forward(self, x):
        h = x.view(-1, 1)
        h_rec = self.ann(h)
        h_rec = h_rec.view(x.size()[0], 12, -1)

        X_rec_1 = h_rec[:, 0, :].view(x.size()[0], 1, -1)
        X_rec_2 = h_rec[:, 1, :].view(x.size()[0], 1, -1)
        X_rec_3 = h_rec[:, 2, :].view(x.size()[0], 1, -1)
        X_rec_avr = h_rec[:, 3, :].view(x.size()[0], 1, -1)
        X_rec_avl = h_rec[:, 4, :].view(x.size()[0], 1, -1)
        X_rec_avf = h_rec[:, 5, :].view(x.size()[0], 1, -1)
        X_rec_v1 = h_rec[:, 6, :].view(x.size()[0], 1, -1)
        X_rec_v2 = h_rec[:, 7, :].view(x.size()[0], 1, -1)
        X_rec_v3 = h_rec[:, 8, :].view(x.size()[0], 1, -1)
        X_rec_v4 = h_rec[:, 9, :].view(x.size()[0], 1, -1)
        X_rec_v5 = h_rec[:, 10, :].view(x.size()[0], 1, -1)
        X_rec_v6 = h_rec[:, 11, :].view(x.size()[0], 1, -1)
        return [X_rec_1, X_rec_2, X_rec_3, X_rec_avr, X_rec_avl, X_rec_avf, X_rec_v1, X_rec_v2, X_rec_v3, X_rec_v4, X_rec_v5, X_rec_v6]

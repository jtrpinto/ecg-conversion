'''
Script to evaluate the proposed model for interlead conversion.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''


import sys
import pickle as pk
import numpy as np
import torch

from datasets import Dataset
from models import UNet
from losses import RMSELoss, AE_loss, corr
from skimage.metrics import structural_similarity as ssim


MODEL = "path_to_model"     # where the trained model was saved

SHARED_ENC = True              # whether or not to use a shared encoder (must match saved model)
LEAD_IN = 'I'                   # input lead

DATABASE = 'PTB'                # which database to use (PTB, INCART, or PTB-XL)
FS = 1000.0                     # data sampling frequency                 
DATASET = 'test'                # which dataset to use (train or test)


# Use GPU if one is available
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


dataset = Dataset(FS, database=DATABASE, lead_in=LEAD_IN, dataset=DATASET)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)


print('\n ======= TESTING MODEL UNET ' + MODEL + ' ======= \n')

model = UNet(shared_enc=SHARED_ENC, bilinear=True).to(DEVICE)
model.load_state_dict(torch.load(MODEL + '.pth', map_location=DEVICE))

loss_fn = RMSELoss().to(DEVICE)

leads_list = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

lead_i_original = list()
lead_i_reconstructed = list()
keys_reconstructed = list()

with torch.no_grad():
    model.eval()  # inference mode

    # compute testing loss
    test_rmse = list()
    test_r = list()
    test_dtw = list()
    test_ssim = list()
    
    for i, (X_in, X_lead1, X_lead2, X_lead3, X_lead_avr, X_lead_avl, X_lead_avf, X_lead_v1, X_lead_v2, X_lead_v3, X_lead_v4, X_lead_v5, X_lead_v6, key) in enumerate(test_loader):
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
        key = key.float()

        X_target = [X_lead1, X_lead2, X_lead3,
                    X_lead_avr, X_lead_avl, X_lead_avf,
                    X_lead_v1, X_lead_v2, X_lead_v3,
                    X_lead_v4, X_lead_v5, X_lead_v6]

        X_rec = model(X_in)

        test_rmse.append([float(rms.detach().cpu().numpy()) for rms in AE_loss(X_rec, X_target, loss_fn, reduce=False)])
        test_r.append([float(rr.detach().cpu().numpy()) for rr in corr(X_rec, X_target, reduce=False)])

        X_target_list = [tt.detach().cpu().numpy()[0, 0, :] for tt in X_target]
        X_rec_list = [zz.detach().cpu().numpy()[0, 0, :] for zz in X_rec]

        lead_i_original.append(X_target_list[0])
        lead_i_reconstructed.append(X_rec_list[0])
        keys_reconstructed.append(key.cpu().numpy())

        # Compute SSIM:
        test_ssim.append([ssim(ss[0], ss[1], data_range=ss[0].max()-ss[0].min()) for ss in zip(X_target_list, X_rec_list)])

        sys.stdout.write('\r' + '........test sample {} / {} '.format(i + 1, len(dataset)))
        sys.stdout.flush()


test_rmse = np.array(test_rmse)
test_r = np.array(test_r)
test_dtw = np.array(test_dtw)
test_ssim = np.array(test_ssim)

print()
print('RMSE', np.mean(test_rmse, axis=0))
print('R (avg)', np.mean(test_r, axis=0))
print('R (med)', np.median(test_r, axis=0))
print('SSIM', np.mean(test_ssim, axis=0))
print()

# Save results to a pickle:
with open(MODEL + '_test_results_' + DATABASE + '.pk', 'wb') as hf:
    pk.dump((test_rmse, test_r, test_dtw, test_ssim), hf)

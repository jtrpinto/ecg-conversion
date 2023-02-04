'''
Implementation of the interlead conversion model training loop.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''

import sys
import torch
import pickle
import numpy as np 

from losses import AE_loss, corr


def train_model(model, loss_fn, optimizer, train_loader, n_epochs, valid_loader=None, save_file='model', device='cpu', patience=np.inf):
    train_hist = []
    train_idr = []
    valid_hist = []
    valid_idr = []

    # For early stopping:
    plateau = 0  
    best_valid_loss = None
    
    # Repeat cycle for the desired maximum number of epochs:
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))

        # training phase
        model.train()  # training mode
        
        for i, (X_in, X_lead1, X_lead2, X_lead3, X_lead_avr, X_lead_avl, X_lead_avf, X_lead_v1, X_lead_v2, X_lead_v3, X_lead_v4, X_lead_v5, X_lead_v6, _) in enumerate(train_loader):
            X_in = X_in.float().to(device)
            X_lead1 = X_lead1.float().to(device)
            X_lead2 = X_lead2.float().to(device)
            X_lead3 = X_lead3.float().to(device)
            X_lead_avr = X_lead_avr.float().to(device)
            X_lead_avl = X_lead_avl.float().to(device)
            X_lead_avf = X_lead_avf.float().to(device)
            X_lead_v1 = X_lead_v1.float().to(device)
            X_lead_v2 = X_lead_v2.float().to(device)
            X_lead_v3 = X_lead_v3.float().to(device)
            X_lead_v4 = X_lead_v4.float().to(device)
            X_lead_v5 = X_lead_v5.float().to(device)
            X_lead_v6 = X_lead_v6.float().to(device)

            X_target = [X_lead1, X_lead2, X_lead3,
                        X_lead_avr, X_lead_avl, X_lead_avf,
                        X_lead_v1, X_lead_v2, X_lead_v3,
                        X_lead_v4, X_lead_v5, X_lead_v6]
                
            # forward pass
            X_rec = model(X_in)
           
            loss = AE_loss(X_rec, X_target, loss_fn)
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()        # compute new gradients
            optimizer.step()       # optimize the parameters

            # display the mini-batch loss
            sys.stdout.write('\r' + '........{} mini-batch loss: {:.3f} |'.format(i + 1, loss.item()))
            sys.stdout.flush()

            if torch.isnan(loss):
                print('NaN loss. Terminating train.')
                return [], []

        # evaluation phase
        print()
        with torch.no_grad():
            model.eval()  # inference mode

            # compute training loss
            train_loss = 0.
            train_r = 0.
            train_r_list = list()

            for i, (X_in, X_lead1, X_lead2, X_lead3, X_lead_avr, X_lead_avl, X_lead_avf, X_lead_v1, X_lead_v2, X_lead_v3, X_lead_v4, X_lead_v5, X_lead_v6, _) in enumerate(train_loader):
                X_in = X_in.float().to(device)
                X_lead1 = X_lead1.float().to(device)
                X_lead2 = X_lead2.float().to(device)
                X_lead3 = X_lead3.float().to(device)
                X_lead_avr = X_lead_avr.float().to(device)
                X_lead_avl = X_lead_avl.float().to(device)
                X_lead_avf = X_lead_avf.float().to(device)
                X_lead_v1 = X_lead_v1.float().to(device)
                X_lead_v2 = X_lead_v2.float().to(device)
                X_lead_v3 = X_lead_v3.float().to(device)
                X_lead_v4 = X_lead_v4.float().to(device)
                X_lead_v5 = X_lead_v5.float().to(device)
                X_lead_v6 = X_lead_v6.float().to(device)
                
                X_target = [X_lead1, X_lead2, X_lead3,
                            X_lead_avr, X_lead_avl, X_lead_avf,
                            X_lead_v1, X_lead_v2, X_lead_v3,
                            X_lead_v4, X_lead_v5, X_lead_v6]
                    
                X_rec = model(X_in)

                train_loss += AE_loss(X_rec, X_target, loss_fn)
                train_r += corr(X_rec, X_target)
                train_r_list.append(corr(X_rec, X_target).detach().cpu().numpy())
            train_loss /= i + 1
            train_hist.append(train_loss.item())
            print('....train loss : {:.3f}'.format(train_loss.item()))
            
            train_r /= i + 1
            print('....train r = {:.3f}'.format(train_r.item()))
            print()

            if valid_loader is None:
                print()
            else:  # compute validation loss
                valid_loss = 0.
                valid_r = 0
                valid_r_list = list()

                for i, (X_in, X_lead1, X_lead2, X_lead3, X_lead_avr, X_lead_avl, X_lead_avf, X_lead_v1, X_lead_v2, X_lead_v3, X_lead_v4, X_lead_v5, X_lead_v6, _) in enumerate(valid_loader):
                    X_in = X_in.float().to(device)
                    X_lead1 = X_lead1.float().to(device)
                    X_lead2 = X_lead2.float().to(device)
                    X_lead3 = X_lead3.float().to(device)
                    X_lead_avr = X_lead_avr.float().to(device)
                    X_lead_avl = X_lead_avl.float().to(device)
                    X_lead_avf = X_lead_avf.float().to(device)
                    X_lead_v1 = X_lead_v1.float().to(device)
                    X_lead_v2 = X_lead_v2.float().to(device)
                    X_lead_v3 = X_lead_v3.float().to(device)
                    X_lead_v4 = X_lead_v4.float().to(device)
                    X_lead_v5 = X_lead_v5.float().to(device)
                    X_lead_v6 = X_lead_v6.float().to(device)
                
                    X_target = [X_lead1, X_lead2, X_lead3,
                                X_lead_avr, X_lead_avl, X_lead_avf,
                                X_lead_v1, X_lead_v2, X_lead_v3,
                                X_lead_v4, X_lead_v5, X_lead_v6]
                        
                    X_rec = model(X_in)
                    
                    valid_loss += AE_loss(X_rec,X_target, loss_fn)
                    valid_r += corr(X_rec,X_target)
                    valid_r_list.append(corr(X_rec,X_target).detach().cpu().numpy())

                valid_loss /= i + 1
                valid_hist.append(valid_loss.item())
                print('....valid loss = {:.3f}'.format(valid_loss.item()))
                valid_r /= i + 1
                print('....valid r = {:.3f}'.format(valid_r.item()))
                print()

                if best_valid_loss is None:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), save_file + '.pth')

                    with open(save_file + '_trainhist.pk', 'wb') as hf:
                        pickle.dump({'loss': train_hist, 'idr': train_idr}, hf)
                    with open(save_file + '_validhist.pk', 'wb') as hf:
                        pickle.dump({'loss': valid_hist, 'idr': valid_idr}, hf)
                    print('....Saving...')
                elif valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), save_file + '.pth')
                    with open(save_file + '_trainhist.pk', 'wb') as hf:
                        pickle.dump({'loss': train_hist, 'idr': train_idr}, hf)
                    with open(save_file + '_validhist.pk', 'wb') as hf:
                        pickle.dump({'loss': valid_hist, 'idr': valid_idr}, hf)
                    plateau = 0
                    print('....Saving...')
                else:
                    plateau += 1
                    if plateau >= patience:
                        print('....Early stopping the train.')
                        return train_hist, valid_hist, X_rec, X_target

    return train_hist, valid_hist, X_rec, X_target
# ecg-conversion

Code for interlead conversion of ECG signals.

Published in:
- Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso, "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
*BMC Medical Informatics and Decision Making*, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6


## How to use it?

Use train.py and test.py, respectively, to train and, then, to evaluate a given model. Do not forget to adjust any model/training/scenario parameter you wish in the beginning of each 

We used the PTB, INCART, and PTB-XL databases, all available on Physionet, but you can apply this to any ECG data. Just make sure, for your database, you have a pickle file for each lead with X_train and X_test (the numpy arrays containing that lead's signals to be reconstructed). It can also have y_train and y_test with additional useful information (such as diagnosis classes). Make also sure to check datasets.py and adapt as needed for your databases.

The proposed model is implemented in models.py, is based on a U-Net, and uses part of the code from Pytorch U-Net (available at https://github.com/milesial/Pytorch-UNet). The losses used are implemented in losses.py and the training loop is implemented in trainers.py, if you wish to change any of these. 

The baseline approach mentioned in our paper (above) is implemented in baseline_model.py, and can be trained using train_baseline.py and evaluated with test_baseline.py.

The code for the experiments on medical diagnosis with reconstructed signals is inside the diagnosis_experiment folder. 



## Contact:

Any questions, do not hesitate to contact João Ribeiro Pinto at jtpinto\[at\]fe.up.pt





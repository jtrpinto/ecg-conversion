'''
Implementation of the proposed model.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''


from torch import nn
from unet_parts import *


class UNet(nn.Module):
    def __init__(self, shared_enc=True, bilinear=True):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.shared = shared_enc

        N = 4
        factor = 2 if bilinear else 1
        
        # Building the encoder(s):
        if self.shared:
            self.inc = DoubleConv(1, 8*N)                   # 8
            self.down1 = Down(8*N, 16*N)                    # (8, 16)
            self.down2 = Down(16*N, 32*N)                   # (16, 32)
            self.down3 = Down(32*N, 64*N)                   # (32, 64)
            self.down4 = Down(64*N, 128*N // factor)        # (64, 128 // factor)

        else:
            self.inc_1 = DoubleConv(1, 8*N)                   # 8
            self.down1_1 = Down(8*N, 16*N)                    # (8, 16)
            self.down2_1 = Down(16*N, 32*N)                   # (16, 32)
            self.down3_1 = Down(32*N, 64*N)                   # (32, 64)
            self.down4_1 = Down(64*N, 128*N // factor)        # (64, 128 // factor)

            self.inc_2 = DoubleConv(1, 8*N)                   # 8
            self.down1_2 = Down(8*N, 16*N)                    # (8, 16)
            self.down2_2 = Down(16*N, 32*N)                   # (16, 32)
            self.down3_2 = Down(32*N, 64*N)                   # (32, 64)
            self.down4_2 = Down(64*N, 128*N // factor)        # (64, 128 // factor)

            self.inc_3 = DoubleConv(1, 8*N)                   # 8
            self.down1_3 = Down(8*N, 16*N)                    # (8, 16)
            self.down2_3 = Down(16*N, 32*N)                   # (16, 32)
            self.down3_3 = Down(32*N, 64*N)                   # (32, 64)
            self.down4_3 = Down(64*N, 128*N // factor)        # (64, 128 // factor)

            self.inc_avr = DoubleConv(1, 8*N)                   # 8
            self.down1_avr = Down(8*N, 16*N)                    # (8, 16)
            self.down2_avr = Down(16*N, 32*N)                   # (16, 32)
            self.down3_avr = Down(32*N, 64*N)                   # (32, 64)
            self.down4_avr = Down(64*N, 128*N // factor)        # (64, 128 // factor)

            self.inc_avl = DoubleConv(1, 8*N)                   # 8
            self.down1_avl = Down(8*N, 16*N)                    # (8, 16)
            self.down2_avl = Down(16*N, 32*N)                   # (16, 32)
            self.down3_avl = Down(32*N, 64*N)                   # (32, 64)
            self.down4_avl = Down(64*N, 128*N // factor)        # (64, 128 // factor)

            self.inc_avf = DoubleConv(1, 8*N)                   # 8
            self.down1_avf = Down(8*N, 16*N)                    # (8, 16)
            self.down2_avf = Down(16*N, 32*N)                   # (16, 32)
            self.down3_avf = Down(32*N, 64*N)                   # (32, 64)
            self.down4_avf = Down(64*N, 128*N // factor)        # (64, 128 // factor)

            self.inc_v1 = DoubleConv(1, 8*N)                   # 8
            self.down1_v1 = Down(8*N, 16*N)                    # (8, 16)
            self.down2_v1 = Down(16*N, 32*N)                   # (16, 32)
            self.down3_v1 = Down(32*N, 64*N)                   # (32, 64)
            self.down4_v1 = Down(64*N, 128*N // factor)        # (64, 128 // factor)

            self.inc_v2 = DoubleConv(1, 8*N)                   # 8
            self.down1_v2 = Down(8*N, 16*N)                    # (8, 16)
            self.down2_v2 = Down(16*N, 32*N)                   # (16, 32)
            self.down3_v2 = Down(32*N, 64*N)                   # (32, 64)
            self.down4_v2 = Down(64*N, 128*N // factor)        # (64, 128 // factor)

            self.inc_v3 = DoubleConv(1, 8*N)                   # 8
            self.down1_v3 = Down(8*N, 16*N)                    # (8, 16)
            self.down2_v3 = Down(16*N, 32*N)                   # (16, 32)
            self.down3_v3 = Down(32*N, 64*N)                   # (32, 64)
            self.down4_v3 = Down(64*N, 128*N // factor)        # (64, 128 // factor)

            self.inc_v4 = DoubleConv(1, 8*N)                   # 8
            self.down1_v4 = Down(8*N, 16*N)                    # (8, 16)
            self.down2_v4 = Down(16*N, 32*N)                   # (16, 32)
            self.down3_v4 = Down(32*N, 64*N)                   # (32, 64)
            self.down4_v4 = Down(64*N, 128*N // factor)        # (64, 128 // factor)

            self.inc_v5 = DoubleConv(1, 8*N)                   # 8
            self.down1_v5 = Down(8*N, 16*N)                    # (8, 16)
            self.down2_v5 = Down(16*N, 32*N)                   # (16, 32)
            self.down3_v5 = Down(32*N, 64*N)                   # (32, 64)
            self.down4_v5 = Down(64*N, 128*N // factor)        # (64, 128 // factor)

            self.inc_v6 = DoubleConv(1, 8*N)                   # 8
            self.down1_v6 = Down(8*N, 16*N)                    # (8, 16)
            self.down2_v6 = Down(16*N, 32*N)                   # (16, 32)
            self.down3_v6 = Down(32*N, 64*N)                   # (32, 64)
            self.down4_v6 = Down(64*N, 128*N // factor)        # (64, 128 // factor)


        # Building the decoders:
        self.up1_1 = Up(128*N, 64*N // factor, bilinear)    # (128, 64 // factor, bilinear=True)
        self.up2_1 = Up(64*N, 32*N // factor, bilinear)     # (64, 32 // factor, bilinear=True)
        self.up3_1 = Up(32*N, 16*N // factor, bilinear)     # (32, 16 // factor, bilinear=True)
        self.up4_1 = Up(16*N, 8*N, bilinear)                # (16, 8, bilinear=True)
        self.outc_1 = OutConv(8*N, 1)                       # (8, n_classes)

        self.up1_2 = Up(128*N, 64*N // factor, bilinear)    # (128, 64 // factor, bilinear=True)
        self.up2_2 = Up(64*N, 32*N // factor, bilinear)     # (64, 32 // factor, bilinear=True)
        self.up3_2 = Up(32*N, 16*N // factor, bilinear)     # (32, 16 // factor, bilinear=True)
        self.up4_2 = Up(16*N, 8*N, bilinear)                # (16, 8, bilinear=True)
        self.outc_2 = OutConv(8*N, 1)                       # (8, n_classes)

        self.up1_3 = Up(128*N, 64*N // factor, bilinear)    # (128, 64 // factor, bilinear=True)
        self.up2_3 = Up(64*N, 32*N // factor, bilinear)     # (64, 32 // factor, bilinear=True)
        self.up3_3 = Up(32*N, 16*N // factor, bilinear)     # (32, 16 // factor, bilinear=True)
        self.up4_3 = Up(16*N, 8*N, bilinear)                # (16, 8, bilinear=True)
        self.outc_3 = OutConv(8*N, 1)                       # (8, n_classes)

        self.up1_avr = Up(128*N, 64*N // factor, bilinear)  # (128, 64 // factor, bilinear=True)
        self.up2_avr = Up(64*N, 32*N // factor, bilinear)   # (64, 32 // factor, bilinear=True)
        self.up3_avr = Up(32*N, 16*N // factor, bilinear)   # (32, 16 // factor, bilinear=True)
        self.up4_avr = Up(16*N, 8*N, bilinear)              # (16, 8, bilinear=True)
        self.outc_avr = OutConv(8*N, 1)                     # (8, n_classes)

        self.up1_avf = Up(128*N, 64*N // factor, bilinear)  # (128, 64 // factor, bilinear=True)
        self.up2_avf = Up(64*N, 32*N // factor, bilinear)   # (64, 32 // factor, bilinear=True)
        self.up3_avf = Up(32*N, 16*N // factor, bilinear)   # (32, 16 // factor, bilinear=True)
        self.up4_avf = Up(16*N, 8*N, bilinear)              # (16, 8, bilinear=True)
        self.outc_avf = OutConv(8*N, 1)                     # (8, n_classes)

        self.up1_avl = Up(128*N, 64*N // factor, bilinear)  # (128, 64 // factor, bilinear=True)
        self.up2_avl = Up(64*N, 32*N // factor, bilinear)   # (64, 32 // factor, bilinear=True)
        self.up3_avl = Up(32*N, 16*N // factor, bilinear)   # (32, 16 // factor, bilinear=True)
        self.up4_avl = Up(16*N, 8*N, bilinear)              # (16, 8, bilinear=True)
        self.outc_avl = OutConv(8*N, 1)                     # (8, n_classes)

        self.up1_v1 = Up(128*N, 64*N // factor, bilinear)   # (128, 64 // factor, bilinear=True)
        self.up2_v1 = Up(64*N, 32*N // factor, bilinear)    # (64, 32 // factor, bilinear=True)
        self.up3_v1 = Up(32*N, 16*N // factor, bilinear)    # (32, 16 // factor, bilinear=True)
        self.up4_v1 = Up(16*N, 8*N, bilinear)               # (16, 8, bilinear=True)
        self.outc_v1 = OutConv(8*N, 1)                      # (8, n_classes)

        self.up1_v2 = Up(128*N, 64*N // factor, bilinear)   # (128, 64 // factor, bilinear=True)
        self.up2_v2 = Up(64*N, 32*N // factor, bilinear)    # (64, 32 // factor, bilinear=True)
        self.up3_v2 = Up(32*N, 16*N // factor, bilinear)    # (32, 16 // factor, bilinear=True)
        self.up4_v2 = Up(16*N, 8*N, bilinear)               # (16, 8, bilinear=True)
        self.outc_v2 = OutConv(8*N, 1)                      # (8, n_classes)

        self.up1_v3 = Up(128*N, 64*N // factor, bilinear)   # (128, 64 // factor, bilinear=True)
        self.up2_v3 = Up(64*N, 32*N // factor, bilinear)    # (64, 32 // factor, bilinear=True)
        self.up3_v3 = Up(32*N, 16*N // factor, bilinear)    # (32, 16 // factor, bilinear=True)
        self.up4_v3 = Up(16*N, 8*N, bilinear)               # (16, 8, bilinear=True)
        self.outc_v3 = OutConv(8*N, 1)                      # (8, n_classes)

        self.up1_v4 = Up(128*N, 64*N // factor, bilinear)   # (128, 64 // factor, bilinear=True)
        self.up2_v4 = Up(64*N, 32*N // factor, bilinear)    # (64, 32 // factor, bilinear=True)
        self.up3_v4 = Up(32*N, 16*N // factor, bilinear)    # (32, 16 // factor, bilinear=True)
        self.up4_v4 = Up(16*N, 8*N, bilinear)               # (16, 8, bilinear=True)
        self.outc_v4 = OutConv(8*N, 1)                      # (8, n_classes)

        self.up1_v5 = Up(128*N, 64*N // factor, bilinear)   # (128, 64 // factor, bilinear=True)
        self.up2_v5 = Up(64*N, 32*N // factor, bilinear)    # (64, 32 // factor, bilinear=True)
        self.up3_v5 = Up(32*N, 16*N // factor, bilinear)    # (32, 16 // factor, bilinear=True)
        self.up4_v5 = Up(16*N, 8*N, bilinear)               # (16, 8, bilinear=True)
        self.outc_v5 = OutConv(8*N, 1)                      # (8, n_classes)

        self.up1_v6 = Up(128*N, 64*N // factor, bilinear)   # (128, 64 // factor, bilinear=True)
        self.up2_v6 = Up(64*N, 32*N // factor, bilinear)    # (64, 32 // factor, bilinear=True)
        self.up3_v6 = Up(32*N, 16*N // factor, bilinear)    # (32, 16 // factor, bilinear=True)
        self.up4_v6 = Up(16*N, 8*N, bilinear)               # (16, 8, bilinear=True)
        self.outc_v6 = OutConv(8*N, 1)                      # (8, n_classes)


    def forward(self, x):
        if self.shared:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            X_rec_1 = self.up1_1(x5, x4)
            X_rec_1 = self.up2_1(X_rec_1, x3)
            X_rec_1 = self.up3_1(X_rec_1, x2)
            X_rec_1 = self.up4_1(X_rec_1, x1)
            X_rec_1 = self.outc_1(X_rec_1)

            X_rec_2 = self.up1_2(x5, x4)
            X_rec_2 = self.up2_2(X_rec_2, x3)
            X_rec_2 = self.up3_2(X_rec_2, x2)
            X_rec_2 = self.up4_2(X_rec_2, x1)
            X_rec_2 = self.outc_2(X_rec_2)

            X_rec_3 = self.up1_3(x5, x4)
            X_rec_3 = self.up2_3(X_rec_3, x3)
            X_rec_3 = self.up3_3(X_rec_3, x2)
            X_rec_3 = self.up4_3(X_rec_3, x1)
            X_rec_3 = self.outc_3(X_rec_3)

            X_rec_avr = self.up1_avr(x5, x4)
            X_rec_avr = self.up2_avr(X_rec_avr, x3)
            X_rec_avr = self.up3_avr(X_rec_avr, x2)
            X_rec_avr = self.up4_avr(X_rec_avr, x1)
            X_rec_avr = self.outc_avr(X_rec_avr)

            X_rec_avf = self.up1_avf(x5, x4)
            X_rec_avf = self.up2_avf(X_rec_avf, x3)
            X_rec_avf = self.up3_avf(X_rec_avf, x2)
            X_rec_avf = self.up4_avf(X_rec_avf, x1)
            X_rec_avf = self.outc_avf(X_rec_avf)

            X_rec_avl = self.up1_avl(x5, x4)
            X_rec_avl = self.up2_avl(X_rec_avl, x3)
            X_rec_avl = self.up3_avl(X_rec_avl, x2)
            X_rec_avl = self.up4_avl(X_rec_avl, x1)
            X_rec_avl = self.outc_avl(X_rec_avl)

            X_rec_v1 = self.up1_v1(x5, x4)
            X_rec_v1 = self.up2_v1(X_rec_v1, x3)
            X_rec_v1 = self.up3_v1(X_rec_v1, x2)
            X_rec_v1 = self.up4_v1(X_rec_v1, x1)
            X_rec_v1 = self.outc_v1(X_rec_v1)

            X_rec_v2 = self.up1_v2(x5, x4)
            X_rec_v2 = self.up2_v2(X_rec_v2, x3)
            X_rec_v2 = self.up3_v2(X_rec_v2, x2)
            X_rec_v2 = self.up4_v2(X_rec_v2, x1)
            X_rec_v2 = self.outc_v2(X_rec_v2)

            X_rec_v3 = self.up1_v3(x5, x4)
            X_rec_v3 = self.up2_v3(X_rec_v3, x3)
            X_rec_v3 = self.up3_v3(X_rec_v3, x2)
            X_rec_v3 = self.up4_v3(X_rec_v3, x1)
            X_rec_v3 = self.outc_v3(X_rec_v3)

            X_rec_v4 = self.up1_v4(x5, x4)
            X_rec_v4 = self.up2_v4(X_rec_v4, x3)
            X_rec_v4 = self.up3_v4(X_rec_v4, x2)
            X_rec_v4 = self.up4_v4(X_rec_v4, x1)
            X_rec_v4 = self.outc_v4(X_rec_v4)

            X_rec_v5 = self.up1_v5(x5, x4)
            X_rec_v5 = self.up2_v5(X_rec_v5, x3)
            X_rec_v5 = self.up3_v5(X_rec_v5, x2)
            X_rec_v5 = self.up4_v5(X_rec_v5, x1)
            X_rec_v5 = self.outc_v5(X_rec_v5)

            X_rec_v6 = self.up1_v6(x5, x4)
            X_rec_v6 = self.up2_v6(X_rec_v6, x3)
            X_rec_v6 = self.up3_v6(X_rec_v6, x2)
            X_rec_v6 = self.up4_v6(X_rec_v6, x1)
            X_rec_v6 = self.outc_v6(X_rec_v6)

            return [X_rec_1, X_rec_2, X_rec_3, X_rec_avr, X_rec_avl, X_rec_avf, X_rec_v1, X_rec_v2, X_rec_v3, X_rec_v4, X_rec_v5, X_rec_v6]
        
        else:
            x1_1 = self.inc_1(x)
            x2_1 = self.down1_1(x1_1)
            x3_1 = self.down2_1(x2_1)
            x4_1 = self.down3_1(x3_1)
            x5_1 = self.down4_1(x4_1)

            x1_2 = self.inc_2(x)
            x2_2 = self.down1_2(x1_2)
            x3_2 = self.down2_2(x2_2)
            x4_2 = self.down3_2(x3_2)
            x5_2 = self.down4_2(x4_2)

            x1_3 = self.inc_3(x)
            x2_3 = self.down1_3(x1_3)
            x3_3 = self.down2_3(x2_3)
            x4_3 = self.down3_3(x3_3)
            x5_3 = self.down4_3(x4_3)

            x1_avr = self.inc_avr(x)
            x2_avr = self.down1_avr(x1_avr)
            x3_avr = self.down2_avr(x2_avr)
            x4_avr = self.down3_avr(x3_avr)
            x5_avr = self.down4_avr(x4_avr)

            x1_avl = self.inc_avl(x)
            x2_avl = self.down1_avl(x1_avl)
            x3_avl = self.down2_avl(x2_avl)
            x4_avl = self.down3_avl(x3_avl)
            x5_avl = self.down4_avl(x4_avl)

            x1_avf = self.inc_avf(x)
            x2_avf = self.down1_avf(x1_avf)
            x3_avf = self.down2_avf(x2_avf)
            x4_avf = self.down3_avf(x3_avf)
            x5_avf = self.down4_avf(x4_avf)

            x1_v1 = self.inc_v1(x)
            x2_v1 = self.down1_v1(x1_v1)
            x3_v1 = self.down2_v1(x2_v1)
            x4_v1 = self.down3_v1(x3_v1)
            x5_v1 = self.down4_v1(x4_v1)

            x1_v2 = self.inc_v2(x)
            x2_v2 = self.down1_v2(x1_v2)
            x3_v2 = self.down2_v2(x2_v2)
            x4_v2 = self.down3_v2(x3_v2)
            x5_v2 = self.down4_v2(x4_v2)

            x1_v3 = self.inc_v3(x)
            x2_v3 = self.down1_v3(x1_v3)
            x3_v3 = self.down2_v3(x2_v3)
            x4_v3 = self.down3_v3(x3_v3)
            x5_v3 = self.down4_v3(x4_v3)

            x1_v4 = self.inc_v4(x)
            x2_v4 = self.down1_v4(x1_v4)
            x3_v4 = self.down2_v4(x2_v4)
            x4_v4 = self.down3_v4(x3_v4)
            x5_v4 = self.down4_v4(x4_v4)

            x1_v5 = self.inc_v5(x)
            x2_v5 = self.down1_v5(x1_v5)
            x3_v5 = self.down2_v5(x2_v5)
            x4_v5 = self.down3_v5(x3_v5)
            x5_v5 = self.down4_v5(x4_v5)

            x1_v6 = self.inc_v6(x)
            x2_v6 = self.down1_v6(x1_v6)
            x3_v6 = self.down2_v6(x2_v6)
            x4_v6 = self.down3_v6(x3_v6)
            x5_v6 = self.down4_v6(x4_v6)

            X_rec_1 = self.up1_1(x5_1, x4_1)
            X_rec_1 = self.up2_1(X_rec_1, x3_1)
            X_rec_1 = self.up3_1(X_rec_1, x2_1)
            X_rec_1 = self.up4_1(X_rec_1, x1_1)
            X_rec_1 = self.outc_1(X_rec_1)

            X_rec_2 = self.up1_2(x5_2, x4_2)
            X_rec_2 = self.up2_2(X_rec_2, x3_2)
            X_rec_2 = self.up3_2(X_rec_2, x2_2)
            X_rec_2 = self.up4_2(X_rec_2, x1_2)
            X_rec_2 = self.outc_2(X_rec_2)

            X_rec_3 = self.up1_3(x5_3, x4_3)
            X_rec_3 = self.up2_3(X_rec_3, x3_3)
            X_rec_3 = self.up3_3(X_rec_3, x2_3)
            X_rec_3 = self.up4_3(X_rec_3, x1_3)
            X_rec_3 = self.outc_3(X_rec_3)

            X_rec_avr = self.up1_avr(x5_avr, x4_avr)
            X_rec_avr = self.up2_avr(X_rec_avr, x3_avr)
            X_rec_avr = self.up3_avr(X_rec_avr, x2_avr)
            X_rec_avr = self.up4_avr(X_rec_avr, x1_avr)
            X_rec_avr = self.outc_avr(X_rec_avr)

            X_rec_avf = self.up1_avf(x5_avf, x4_avf)
            X_rec_avf = self.up2_avf(X_rec_avf, x3_avf)
            X_rec_avf = self.up3_avf(X_rec_avf, x2_avf)
            X_rec_avf = self.up4_avf(X_rec_avf, x1_avf)
            X_rec_avf = self.outc_avf(X_rec_avf)

            X_rec_avl = self.up1_avl(x5_avl, x4_avl)
            X_rec_avl = self.up2_avl(X_rec_avl, x3_avl)
            X_rec_avl = self.up3_avl(X_rec_avl, x2_avl)
            X_rec_avl = self.up4_avl(X_rec_avl, x1_avl)
            X_rec_avl = self.outc_avl(X_rec_avl)

            X_rec_v1 = self.up1_v1(x5_v1, x4_v1)
            X_rec_v1 = self.up2_v1(X_rec_v1, x3_v1)
            X_rec_v1 = self.up3_v1(X_rec_v1, x2_v1)
            X_rec_v1 = self.up4_v1(X_rec_v1, x1_v1)
            X_rec_v1 = self.outc_v1(X_rec_v1)

            X_rec_v2 = self.up1_v2(x5_v2, x4_v2)
            X_rec_v2 = self.up2_v2(X_rec_v2, x3_v2)
            X_rec_v2 = self.up3_v2(X_rec_v2, x2_v2)
            X_rec_v2 = self.up4_v2(X_rec_v2, x1_v2)
            X_rec_v2 = self.outc_v2(X_rec_v2)

            X_rec_v3 = self.up1_v3(x5_v3, x4_v3)
            X_rec_v3 = self.up2_v3(X_rec_v3, x3_v3)
            X_rec_v3 = self.up3_v3(X_rec_v3, x2_v3)
            X_rec_v3 = self.up4_v3(X_rec_v3, x1_v3)
            X_rec_v3 = self.outc_v3(X_rec_v3)

            X_rec_v4 = self.up1_v4(x5_v4, x4_v4)
            X_rec_v4 = self.up2_v4(X_rec_v4, x3_v4)
            X_rec_v4 = self.up3_v4(X_rec_v4, x2_v4)
            X_rec_v4 = self.up4_v4(X_rec_v4, x1_v4)
            X_rec_v4 = self.outc_v4(X_rec_v4)

            X_rec_v5 = self.up1_v5(x5_v5, x4_v5)
            X_rec_v5 = self.up2_v5(X_rec_v5, x3_v5)
            X_rec_v5 = self.up3_v5(X_rec_v5, x2_v5)
            X_rec_v5 = self.up4_v5(X_rec_v5, x1_v5)
            X_rec_v5 = self.outc_v5(X_rec_v5)

            X_rec_v6 = self.up1_v6(x5_v6, x4_v6)
            X_rec_v6 = self.up2_v6(X_rec_v6, x3_v6)
            X_rec_v6 = self.up3_v6(X_rec_v6, x2_v6)
            X_rec_v6 = self.up4_v6(X_rec_v6, x1_v6)
            X_rec_v6 = self.outc_v6(X_rec_v6)

            return [X_rec_1, X_rec_2, X_rec_3, X_rec_avr, X_rec_avl, X_rec_avf, X_rec_v1, X_rec_v2, X_rec_v3, X_rec_v4, X_rec_v5, X_rec_v6]
        

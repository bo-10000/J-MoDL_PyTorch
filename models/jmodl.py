import torch
import torch.nn as nn
import math
from utils import r2c, c2r

#CNN denoiser ======================
def convblock(in_channels, out_channels, kernel_size=3, padding=1, bn=False, relu=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def double_conv(in_channels, out_channels, bn=False):
    return nn.Sequential(
        convblock(in_channels, out_channels, bn=bn),
        convblock(out_channels, out_channels, bn=bn)
    )

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super().__init__()
        self.conv = double_conv(in_channels, out_channels, bn=bn)

    def forward(self, x1, x2):
        x1 = nn.Upsample(size=x2.shape[-2:], mode='bilinear')(x1)
        out = torch.cat((x2, x1), axis=1)
        out = self.conv(out)
        return out


class bigUnet(nn.Module):
    def __init__(self, bn=False):
        super().__init__()
        
        self.down1 = double_conv(2, 64, bn=bn)

        self.down2 = nn.Sequential(
            nn.AvgPool2d(2, 2), double_conv(64, 128, bn=bn)
        )

        self.down3 = nn.Sequential(
            nn.AvgPool2d(2, 2), double_conv(128, 256, bn=bn)
        )

        self.down4 = nn.Sequential(
            nn.AvgPool2d(2, 2), double_conv(256, 512, bn=bn)
        )

        self.center = nn.Sequential(
            nn.AvgPool2d(2, 2), convblock(512, 512, bn=bn)
        )

        self.up1 = UpSample(1024, 256, bn=bn)
        self.up2 = UpSample(512, 128, bn=bn)
        self.up3 = UpSample(256, 64, bn=bn)
        self.up4 = UpSample(128, 64, bn=bn)

        self.final = convblock(64, 2, 1, 0, bn=bn, relu=False)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        out = self.center(x4)
        out = self.up1(out, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.final(out)
        return out

class smallModel(nn.Module):
    def __init__(self, bn=False):
        super().__init__()
        self.layer = nn.Sequential(
            double_conv(2, 64, bn=bn),
            double_conv(64, 128, bn=bn),
            double_conv(128, 64, bn=bn),
            convblock(64, 2, 1, bn=bn, relu=False)
        )

    def forward(self, x):
        return self.layer(x)

#CG algorithm ======================
def myAtA(img, csm, atah, aatv):
    cimg = csm * img
    tmp = atah@cimg@aatv
    coilComb = torch.sum(tmp*csm.conj(), axis=1)
    return coilComb

class data_consistency(nn.Module):
    def __init__(self, cgIter, cgTol, lam):
        super().__init__()
        self.cgIter = cgIter
        self.cgTol = cgTol
        self.lam = lam

    def forward(self, z, atah, aatv, atb, csm):
        z = r2c(z, axis=1)
        rhs = atb + self.lam * z
        
        #cg algorithm
        x = torch.zeros_like(rhs)
        i, r, p = 0, rhs, rhs
        rTr = torch.sum(r.conj()*r).real
        while i < self.cgIter and rTr > self.cgTol:
            Ap = myAtA(p, csm, atah, aatv) + self.lam * p
            alpha = rTr / torch.sum(p.conj()*Ap).real
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = torch.sum(r.conj()*r).real
            beta = rTrNew / rTr
            p = r + beta * p
            i += 1
            rTr = rTrNew

        return c2r(x, axis=1)

#forward model ====================
def getAAt(m, N): #FT + apply mask + IFT
    n = torch.arange(N).to(m.device)
    jTwoPi = 1j*2*math.pi
    scale = 1./(N**0.5)

    A = torch.exp(-jTwoPi*(m-1/2)*(n-N/2))*scale
    At = A.transpose(0, 1).conj()
    return A, At

class At(nn.Module):
    def __init__(self, initx, inity, sigma):
        super().__init__()
        self.kx = nn.Parameter(torch.tensor(initx), requires_grad=True)
        self.ky = nn.Parameter(torch.tensor(inity), requires_grad=True)
        self.sigma = sigma

    def forward(self, x, csm):
        """
        :x: full-sampled image (nrow x ncol) - complex64
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        """
        ncoil, M, N = csm.shape[-3:]
        Ah, Aht = getAAt(self.kx, M)
        Av, Avt = getAAt(self.ky, N)
        Av, Avt = Av.transpose(0, 1), Avt.transpose(0, 1)
        atah = Aht@Ah
        aatv = Av@Avt

        b = Ah@(x.repeat(ncoil, 1, 1, 1).permute(1, 0, 2, 3) * csm)@Av
        noise = torch.normal(0, 1, size=b.shape) + 1j*torch.normal(0, 1, size=b.shape)
        b += self.sigma * noise.to(b.device)
        atb = torch.sum(csm.conj()*(Aht@b@Avt), axis=1)

        return atah, aatv, atb

#model =======================    
class JMoDL(nn.Module):
    def __init__(self, k_iters, lam, initx, inity, sigma, cgIter=10, cgTol=1e-10, denoiser_type='unet', bn=False):
        """
        :k_iters: number of iterations
        :initx, inity: initial sampling positions. size: (num_kspace * acc_rate, 1)
        :denoiser_type: either [unet/cnn]
        :sigma: for noise
        :bn: Whether to use BatchNormalization for cnn denoiser
        """
        super().__init__()
        self.k_iters = k_iters

        self.at = At(initx, inity, sigma)

        cnn_denoiser = bigUnet if denoiser_type=='unet' else smallModel
        self.dw = cnn_denoiser(bn=bn)
        self.dc = data_consistency(cgIter, cgTol, lam)

    def forward(self, gt, csm):
        """
        -parameters-
        :gt: full-sampled image (nrow x ncol) - complex64
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        -returns-
        :abt: undersampled image (nrow x ncol) - complex64
        :x: reconstructed image (nrow x ncol) - complex64
        """
        atah, aatv, atb = self.at(gt, csm)

        z = torch.zeros_like(atb)
        z = c2r(z, axis=1)
        x = self.dc(z, atah, aatv, atb, csm)

        for _ in range(self.k_iters):
            #dw 
            z = self.dw(x) # (2, nrow, ncol)
            #dc
            x = self.dc(z, atah, aatv, atb, csm) # (2, nrow, ncol)

        return atb, r2c(x, axis=1)
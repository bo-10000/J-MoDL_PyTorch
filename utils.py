import math
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

#train utils ==============================
class Logger():
    def __init__(self, log_dir):
        self.log_dir = log_dir
    def write(self, log_message, verbose=True):
        with open(self.log_dir, 'a') as f:
            f.write(log_message)
            f.write('\n')
        if verbose:
            print(log_message)

def set_seeds(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

def get_init_mask(init_mask_path):
    init_mask = np.load(init_mask_path)
    initx = init_mask['kh']
    inity = init_mask['kv']
    initx = np.where(initx==True)[0][:,None].astype(np.float32)/len(initx) #(num_kspace*acc_rate, 1)
    inity = np.where(inity==True)[0][:,None].astype(np.float32)/len(inity) #(num_kspace*acc_rate, 1)
    return initx, inity

def generate_init_mask(N, m, ACS=0.04):
    """
    Acc factor = round(N/m)
    <Params>
    :N: number of total lines
    :m: number of sampling locations, including the ACS lines
    :ACS: ratio of ACS lines (ACS line # = round(N*ACS))
    <Usage>
    :init6x = create_mask(256, 96)
    :init6y = create_mask(232, 101)
    """
    n_ACS = round(N*ACS)
    ACS_start = round(N/2) - round(n_ACS/2)

    a = list(range(0, ACS_start)) + list(range(ACS_start + n_ACS, N))
    maskloc = np.random.choice(a, size=m-n_ACS, replace=False)
    maskloc = np.sort(np.append(maskloc, list(range(ACS_start, ACS_start + n_ACS))))
    
    mask = np.zeros((N, ), dtype=np.bool)
    mask[maskloc] = True
    
    return mask

def clip_mask(model):
    with torch.no_grad():
        model.at.kx.clamp_(0, 1)
        model.at.ky.clamp_(0, 1)

#math ================================
def c2r(complex_img, axis=0):
    """
    :input shape: row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, np.ndarray):
        real_img = np.stack((complex_img.real, complex_img.imag), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        real_img = torch.stack((complex_img.real, complex_img.imag), axis=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(real_img, axis=0):
    """
    :input shape: 2 x row x col (float32)
    :output shape: row x col (complex64)
    """
    if axis == 0:
        complex_img = real_img[0] + 1j*real_img[1]
    elif axis == 1:
        complex_img = real_img[:,0] + 1j*real_img[:,1]
    else:
        raise NotImplementedError
    return complex_img

#metrics ==================================================
def complex_MSE(y_pred, y):
    return torch.mean(torch.pow(torch.abs(y_pred-y), 2))

def psnr_batch(y_batch, y_pred_batch):
    #calculate psnr for every batch and return mean
    mean_psnr = 0
    for batch_idx in range(y_batch.shape[0]):
        y = y_batch[batch_idx]
        y_pred = y_pred_batch[batch_idx]
        mean_psnr += psnr(y, y_pred, y.max())
    return mean_psnr / y_batch.shape[0]

def psnr(y, y_pred, MAX_PIXEL_VALUE=1.0):
    rmse_ = rmse(y, y_pred)
    if rmse_ == 0:
        return float('inf')
    return 20 * math.log10(MAX_PIXEL_VALUE/rmse_+1e-10)

def ssim_batch(y_batch, y_pred_batch):
    mean_ssim = 0
    for batch_idx in range(y_batch.shape[0]):
        y = y_batch[batch_idx]
        y_pred = y_pred_batch[batch_idx]
        mean_ssim += ssim(y, y_pred)
    return mean_ssim / y_batch.shape[0]

def ssim(y, y_pred):
    from skimage.metrics import structural_similarity
    return structural_similarity(y, y_pred)

def mse(y, y_pred):
    return np.mean((y-y_pred)**2)

def rmse(y, y_pred):
    return math.sqrt(mse(y, y_pred))

#display =======================
def get_mask_img(kx, ky, M, N):
    kxF=kx*M
    kyF=ky*N
    kxF=np.int32(kxF)
    kyF=np.int32(kyF)
    mx=np.zeros((M,1),dtype=np.bool)
    mx[kxF]=True
    my=np.zeros((N,1),dtype=np.bool)
    my[kyF]=True
    mask=mx@(my.T)
    return mask

def display_img(x, mask, y, y_pred, score=None):
    fig = plt.figure(figsize=(15,10))
    ax1 = plt.subplot2grid(shape=(2,6), loc=(0,1), colspan=2)
    ax2 = plt.subplot2grid((2,6), (0,3), colspan=2)
    ax3 = plt.subplot2grid((2,6), (1,0), colspan=2)
    ax4 = plt.subplot2grid((2,6), (1,2), colspan=2)
    ax5 = plt.subplot2grid((2,6), (1,4), colspan=2)
    ax1.imshow(x, cmap='gray')
    ax1.set_title('zero-filled')
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('mask')
    ax3.imshow(y, cmap='gray')
    ax3.set_title('GT')
    ax4.imshow(y_pred, cmap='gray')
    ax4.set_title('reconstruction')
    im5 = ax5.imshow(np.abs(y_pred-y)*5, cmap='gray', vmin=np.abs(y).min(), vmax=np.abs(y).max())
    ax5.set_title('diff (x5)')
    fig.colorbar(im5, ax=ax5)
    if score:
        plt.suptitle('score: {:.4f}'.format(score))
    return fig

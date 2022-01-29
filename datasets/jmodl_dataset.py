import torch
from torch.utils.data import Dataset
import numpy as np
import os

class jmodl_dataset(Dataset):
    def __init__(self, mode, dataset_path, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        prefix = 'trn' if mode == 'train' else 'tst'
        dataset_file = os.path.join(dataset_path, prefix+'data_jmodl.npz')

        data = np.load(dataset_file)
        self.gt_all = data[prefix+'Org']
        self.csm_all = data[prefix+'Csm']

        self.sigma = sigma

    def __getitem__(self, index):
        """
        :gt: fully-sampled image (nrow x ncol) - complex64
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        """
        gt = self.gt_all[index]
        csm = self.csm_all[index]
        return torch.from_numpy(gt), torch.from_numpy(csm)

    def __len__(self):
        return len(self.gt_all)
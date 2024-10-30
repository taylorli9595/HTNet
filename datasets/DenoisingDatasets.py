#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:51:11

import torch
import h5py as h5
import random
import cv2
import os
import numpy as np
import torch.utils.data as uData
from skimage import img_as_float32 as img_as_float
from .data_tools import sigma_estimate, random_augmentation, gaussian_kernel
from . import BaseDataSetH5, BaseDataSetImg

# Benchmardk Datasets: Renoir and SIDD
class BenchmarkTrain(BaseDataSetH5):
    def __init__(self, h5_file, length, pch_size=128, L=4):
        super(BenchmarkTrain, self).__init__(h5_file, length)
        self.L = L
        self.sigma_spatial = radius
        self.noise_estimate = noise_estimate
        self.eps2 = eps2
        self.pch_size = pch_size

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[ind_im]]
            im_gt, im_noisy = self.crop_patch(imgs_sets)
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        if self.noise_estimate:
            sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))
        eps2 = torch.tensor([self.eps2], dtype=torch.float32).reshape((1,1,1))

        if self.noise_estimate:
            sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2, 0, 1)))
            return im_noisy, im_gt, sigma2_map_est, eps2
        else:
            return im_noisy, im_gt

class BenchmarkTest(BaseDataSetH5):
    def __getitem__(self, index):
        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            C2 = imgs_sets.shape[2]
            C = int(C2/2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt

# Simulation Datasets:
class SimulateTrain(BaseDataSetImg):
    def __init__(self, im_list, length,  pch_size=128, L=4):
        super(SimulateTrain, self).__init__(im_list, length,  pch_size)
        self.L = L
        self.num_images = len(im_list)

    def __getitem__(self, index):
        pch_size = self.pch_size
        ind_im = random.randint(0, self.num_images-1)

        im_ori = cv2.imread(self.im_list[ind_im], 0)#顺序相反操作,变成RGB
        Img = np.expand_dims(im_ori.copy(), 2)
        #im_ori = np.expand_dims(im_ori[:,:,0], 0)
        im_gt = img_as_float(self.crop_patch(Img))


        # generate noise
        noise_np = np.random.gamma(self.L, 1/self.L,im_gt.shape)
        noise = noise_np.astype(np.float32)

        #noise = torch.from_numpy(noise_np)
        #noise = torch.randn(im_gt.shape).numpy() * sigma_map
        im_noisy = im_gt * noise
        noise = noise + 1e-4
        im_gt, im_noisy, sigmaMapGt = random_augmentation(im_gt, im_noisy, noise)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))
        sigmaMapGt = torch.from_numpy(sigmaMapGt.transpose((2, 0, 1)))

        return im_noisy, im_gt, sigmaMapGt


class SimulateTest(uData.Dataset):
    def __init__(self, im_list, L=4, dep_U=4):
        super(SimulateTest, self).__init__()
        self.im_list = im_list
        self.L = L
        self.dep_U = dep_U
    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        im_ori = cv2.imread(self.im_list[index], 0)#顺序相反操作,变成RGB
        Img = np.expand_dims(im_ori, 2)
        #im_ori = np.expand_dims(im_ori[:,:,0], 0)
        #im_gt = img_as_float(Img)
        H, W, C = im_gt.shape
        #print('H, W, C:',H, W, C)
        H -= int(H % pow(2, self.dep_U))
        W -= int(W % pow(2, self.dep_U))
        #print('H, W:',H, W)
        im_gt = img_as_float(im_gt[:H, :W, ])
        # generate noise
        noise_np = np.random.gamma(self.L, 1/self.L,im_gt.shape)
        noise = noise_np.astype(np.float32)
        im_noisy = im_gt * noise

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))
        sigmaMapGt = torch.from_numpy(noise.transpose((2, 0, 1)))

        return im_noisy, im_gt, sigmaMapGt



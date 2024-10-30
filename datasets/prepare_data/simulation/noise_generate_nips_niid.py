#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-05 14:44:04

import sys
sys.path.append('./')

import numpy as np
import cv2
from skimage import img_as_float
from utils import generate_gauss_kernel_mix, peaks, sincos_kernel
import h5py as h5
from pathlib import Path
import argparse

base_path = Path('test_data')
seed = 10000

np.random.seed(seed)
#tree kernel map
kernels = [peaks(256), sincos_kernel(), generate_gauss_kernel_mix(256, 256)]
#print(list(enumerate(kernels)))
dep_U = 4

sigma_max = 75/255.0
sigma_min = 10/255.0
#for data_name in ['LIVE1_gray', 'Set5_gray', 'BSD68']:

for data_name in ['Set5_gray']:
    if data_name == 'LIVE1_gray' or data_name == 'Set5_gray':
        im_list = sorted((base_path / data_name).glob('*.bmp'))
    else:
        im_list = sorted((base_path / data_name).glob('*.png'))
    im_list = sorted([x.name for x in im_list])

    for jj, sigma in enumerate(kernels):
        print(jj+1)
        print('Case {:d} of Dataset {:s}: {:d} images'.format(jj+1, data_name, len(im_list)))
        print(sigma)
        # generate sigmaMap
        sigma = sigma_min + (sigma-sigma.min())/(sigma.max()-sigma.min()) * (sigma_max-sigma_min)
        #normal?
        print(sigma)
        noise_dir = base_path / 'noise_niid'
        print(noise_dir)
        if not noise_dir.is_dir():
            noise_dir.mkdir()
        h5_path = noise_dir.joinpath(data_name + '_niid_case' + str(jj+1) + '.hdf5')
        print(h5_path)
        if h5_path.exists():
            h5_path.unlink()
        #write content to .h5 file
        with h5.File(h5_path) as h5_file:
            for ii, im_name in enumerate(im_list):
                print(ii)
                gt_name = str(base_path / data_name / im_name)
                print(gt_name)
                im_gt = cv2.imread(gt_name, 1)[:, :, ::-1]
                print(im_gt.shape)
                H, W, C = im_gt.shape
                print('H, W, C:',H, W, C)
                H -= int(H % pow(2, dep_U))
                W -= int(W % pow(2, dep_U))
                print('H, W:',H, W)
                im_gt = img_as_float(im_gt[:H, :W, ])

                sigma = cv2.resize(sigma, (W, H))
                sigma = sigma.astype(np.float32)

                noise = np.random.randn(H, W, C) * np.expand_dims(sigma, 2)
                noise = noise.astype(np.float32)
                data = np.concatenate((noise, sigma[:,:,np.newaxis]), axis=2)
                h5_file.create_dataset(name=im_name.split('.')[0], dtype=data.dtype,
                                                                        shape=data.shape, data=data)


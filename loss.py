#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14

import torch
import torch.nn.functional as F
from math import pi, log
from utils import LogGamma

log_gamma = LogGamma.apply

# clip bound
log_max = log(1e4)
log_min = log(1e-8)
alpha = 1
def loss_fn(out_dncnn, out_unet, im_gt, sigmaMap):
    #C = im_gt.shape[1]
    '''
    Input:
         out_denoise:clean image estimation(f)
         im_gt : clean image
         out_dncnn: clean image estimation(f) from dncnn 
         out_unet: clean image estimation(f) from Unet
         im_noisy:    noisy image(f) 
    '''
    alpha = 0.8
    y = torch.ones(out_dncnn.shape)
    y = y.cuda()
    diff_eta = torch.abs(out_dncnn - im_gt)
    #diff_eta = torch.nn.functional.l1_loss( out_dncnn, sigmaMap, size_average = None, reduce = None, reduction = 'mean')
    L1_alpha = alpha*diff_eta
    a1 = L1_alpha > 1
    a2 = (L1_alpha > 0) & (L1_alpha <= 1)
    #a3 = L1_alpha < -1
    #a4 = (L1_alpha <= 0) & (L1_alpha > -1) 
    Trunc_loss  = a1*torch.log(2+L1_alpha-L1_alpha) - a2*torch.log(1-L1_alpha + 0.5*L1_alpha**2)
    Trunc_loss = torch.mean(Trunc_loss)
    L2_dncnn = torch.nn.functional.mse_loss(out_dncnn, im_gt, size_average = None, reduce = None, reduction = 'mean')
    Loss_mix = 0.8*L2_dncnn +  Trunc_loss
   
    #Trunc_loss  = a1*torch.log(2+L1_alpha-L1_alpha) - a2*torch.log(1+L1_alpha + 0.5*L1_alpha**2) -a3*torch.log(2+L1_alpha-L1_alpha) - a4*torch.log(1 - L1_alpha + 0.5*L1_alpha**2)
    #Trunc_eta = torch.mean(Trunc_eta)


    
    L2_unet = torch.nn.functional.mse_loss(out_unet, im_gt, size_average = None, reduce = None, reduction = 'mean')

    loss =  L2_unet + 0.5*Loss_mix 

    return loss, L2_dncnn, L2_unet, loss, Trunc_loss



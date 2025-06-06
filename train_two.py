#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-10 22:41:49

from glob import glob
import warnings
import time
import random
import numpy as np
import shutil
import torchvision.utils as vutils
from utils import batch_PSNR, batch_SSIM
from tensorboardX import SummaryWriter
from math import ceil
from loss import loss_fn
from networks import VDN, weight_init_kaiming
from datasets import DenoisingDatasets
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
from pathlib import Path
from options import set_opts

# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

args = set_opts()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

if isinstance(args.gpu_id, int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))

_C = 1
_lr_min = 1e-6
_modes = ['train', 'test_bsd68']

def train_model(net, datasets, optimizer, lr_scheduler, criterion):
    clip_grad_D = args.clip_grad_D
    clip_grad_S = args.clip_grad_S
    batch_size = {'train': args.batch_size, 'test_bsd68': 1}
    data_loader = {phase: torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size[phase],
          shuffle=True, num_workers=args.num_workers, pin_memory=True) for phase in datasets.keys()}
    num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in datasets.keys()}
    writer = SummaryWriter(args.log_dir+str(args.L))
    if args.resume:
        step = args.step
        step_img = args.step_img
    else:
        step = 0
        step_img = {x: 0 for x in _modes}
    param_D = [x for name, x in net.named_parameters() if 'dnet' in name.lower()]
    param_S = [x for name, x in net.named_parameters() if 'snet' in name.lower()]
    for epoch in range(args.epoch_start, args.epochs):
        loss_per_epoch = {x: 0 for x in ['Loss', 'L2_dncnn', 'Trunc_loss','L2_unet', 'L2_our']}
        grad_norm_D = grad_norm_S = 0
        tic = time.time()
        # train stage
        net.train()
        # set learning rate
        lr = optimizer.param_groups[0]['lr']
        if lr < _lr_min:
            sys.exit('Reach the minimal learning rate')
        print('learning rate %f' % lr)

        phase = 'train'
        for ii, data in enumerate(data_loader[phase]):
            im_noisy, im_gt, sigmaMapGt = [x.cuda() for x in data]

            optimizer.zero_grad()
            Out_dncnn, Out_unet = net(im_noisy, 'train')
            loss, l2_dncnn, l2_unet, l2_our, trunc_loss = criterion(Out_dncnn, Out_unet, im_gt, sigmaMapGt)
            loss.backward()
            # clip the gradnorm
            total_norm_D = nn.utils.clip_grad_norm_(param_D, clip_grad_D)
            total_norm_S = nn.utils.clip_grad_norm_(param_S, clip_grad_S)
            grad_norm_D = (grad_norm_D*(ii/(ii+1)) + total_norm_D/(ii+1))
            grad_norm_S = (grad_norm_S*(ii/(ii+1)) + total_norm_S/(ii+1))
            optimizer.step()


            loss_per_epoch['Loss'] += loss.item() / num_iter_epoch[phase]
            loss_per_epoch['L2_dncnn'] += l2_dncnn.item() / num_iter_epoch[phase]
            loss_per_epoch['Trunc_loss'] += trunc_loss.item() / num_iter_epoch[phase]
            loss_per_epoch['L2_unet'] += l2_unet.item() / num_iter_epoch[phase]
            loss_per_epoch['L2_our'] += l2_our.item() / num_iter_epoch[phase]
            im_denoise = Out_unet[:, :_C, ].detach().data

            im_denoise.clamp_(0.0, 1.0)

            if (ii+1) % args.print_freq == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, l2_dncnn={:+7.5f}, trunc_loss ={:+>7.5f}, ' + \
                        'L2_unet ={:+>7.5f},l2_our={:+>7.5f}, GNorm_D:{:.1e}/{:.1e}, ' + \
                                                                  'GNorm_S:{:.1e}/{:.1e}, lr={:.1e}'
                print(log_str.format(epoch+1, args.epochs, phase, ii+1, num_iter_epoch[phase],
                                         l2_dncnn.item(),trunc_loss.item(),l2_unet.item(), l2_our.item(), clip_grad_D,
                                                       total_norm_D, clip_grad_S, total_norm_S, lr))
                writer.add_scalar('Train Loss Iter', loss.item(), step)
                writer.add_scalar('Train L2_dncnn Iter', l2_dncnn.item(), step)
                writer.add_scalar('Train L2_unet Iter', l2_unet.item(), step)
                writer.add_scalar('Train L2_our Iter', l2_our.item(), step)
                writer.add_scalar('Train Trunc_loss Iter', trunc_loss.item(), step)
                writer.add_scalar('Gradient Norm_D Iter', total_norm_D, step)
                writer.add_scalar('Gradient Norm_S Iter', total_norm_S, step)
                step += 1
            if (ii+1) % (20*args.print_freq) == 0:
                #sigmaMap_pred = torch.exp(out_sigmaMap)
                x1 = vutils.make_grid(im_denoise, normalize=True, scale_each=True)
                writer.add_image(phase+' Denoised images', x1, step_img[phase])
                x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                x3 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
                writer.add_image(phase+' Noisy Image', x3, step_img[phase])
                step_img[phase] += 1


        log_str ='{:s}: Loss={:+.2e}, l2_dncnn ={:+.5e}, l2_unet={:.5e}, l2_our={:+.5e},trunc_loss ={:+>7.5f},  ' + \
                                                      'GNorm_D={:.1e}/{:.1e}, GNorm_S={:.1e}/{:.1e}'
        print(log_str.format(phase, loss_per_epoch['Loss'], loss_per_epoch['L2_dncnn'],loss_per_epoch['Trunc_loss'],loss_per_epoch['L2_unet'],
                                loss_per_epoch['L2_our'], 
                                                clip_grad_D, grad_norm_D, clip_grad_S, grad_norm_S))
        writer.add_scalar('Loss_epoch', loss_per_epoch['Loss'], epoch)
        writer.add_scalar('Mean Grad Norm_D epoch', grad_norm_D, epoch)
        writer.add_scalar('Mean Grad Norm_S epoch', grad_norm_S, epoch)
        clip_grad_D = min(clip_grad_D, grad_norm_D)
        clip_grad_S = min(clip_grad_S, grad_norm_S)
        print('-'*150)

        # test stage
        net.eval()
        psnr_per_epoch = {x: 0 for x in _modes[1:]}
        ssim_per_epoch = {x: 0 for x in _modes[1:]}
        for phase in _modes[1:]:
            for ii, data in enumerate(data_loader[phase]):
                #im_noisy, im_gt = [x.cuda() for x in data]
                im_noisy, im_gt, sigmaMapGt = [x.cuda() for x in data]

                with torch.set_grad_enabled(False):
                    Out_dncnn, Out_unet = net(im_noisy, 'train')

                im_denoise = torch.clamp(Out_unet[:, :_C, ].detach().data, 0.0, 1.0)
                psnr_iter = batch_PSNR(im_denoise, im_gt)
                ssim_iter = batch_SSIM(im_denoise, im_gt)
                psnr_per_epoch[phase] += psnr_iter
                ssim_per_epoch[phase] += ssim_iter
                # print statistics every log_interval mini_batches
                if (ii+1) % 30 == 0:
                    log_str = '[Epoch:{:>3d}/{:<3d}] {:s}:{:0>5d}/{:0>5d},  ' + \
                        'psnr={:4.2f}, ssim={:5.4f}'
                    print(log_str.format(epoch+1, args.epochs, phase, ii+1, num_iter_epoch[phase],
                                                                          psnr_iter, ssim_iter))
                # tensorboardX summary
                    x1 = vutils.make_grid(im_denoise, normalize=True, scale_each=True)
                    writer.add_image(phase+' Denoised images', x1, step_img[phase])
                    x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                    writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                    x3 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
                    writer.add_image(phase+' Noise Image', x3, step_img[phase])
                    step_img[phase] += 1

            psnr_per_epoch[phase] /= (ii+1)
            ssim_per_epoch[phase] /= (ii+1)
            log_str = '{:s}: PSNR={:4.2f}, SSIM={:5.4f}'
            print(log_str.format(phase, psnr_per_epoch[phase],
                                 ssim_per_epoch[phase]))
            print('-'*90)

        # adjust the learning rate
        lr_scheduler.step()
        # save model
        if (epoch+1) % args.save_model_freq == 0 or epoch+1 == args.epochs:
            model_prefix = 'model_'
            save_path_model = os.path.join(args.model_dir+str(args.L), model_prefix+str(epoch+1))
            torch.save({
                'epoch': epoch+1,
                'step': step+1,
                'step_img': {x: step_img[x] for x in _modes},
                'grad_norm_D': clip_grad_D,
                'grad_norm_S': clip_grad_S,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, save_path_model)
            model_state_prefix = 'model_state_'
            save_path_model_state = os.path.join(args.model_dir+str(args.L), model_state_prefix+str(epoch+1))
            torch.save(net.state_dict(), save_path_model_state)

        #writer.add_scalars('MSE_epoch', mse_per_epoch, epoch)
        writer.add_scalars('PSNR_epoch_test', psnr_per_epoch, epoch)
        writer.add_scalars('SSIM_epoch_test', ssim_per_epoch, epoch)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
    writer.close()
    print('Reach the maximal epochs! Finish training') 

def main():
    # build the model
    net = VDN(_C, slope=args.slope, wf=args.wf, dep_U=args.depth)
    # move the model to GPU
    net = nn.DataParallel(net).cuda()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    args.milestones = [10, 20, 25, 30, 35, 40, 45, 50]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> Loading checkpoint {:s}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.epoch_start = checkpoint['epoch']
            args.step = checkpoint['step']
            args.step_img = checkpoint['step_img']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            net.load_state_dict(checkpoint['model_state_dict'])
            args.clip_grad_D = checkpoint['grad_norm_D']
            args.clip_grad_S = checkpoint['grad_norm_S']
            print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args.resume, checkpoint['epoch']))
        else:
            sys.exit('Please provide corrected model path!')
    else:
        net = weight_init_kaiming(net)
        args.epoch_start = 0
        if os.path.isdir(args.log_dir+str(args.L)):
            shutil.rmtree(args.log_dir+str(args.L))
        os.makedirs(args.log_dir+str(args.L))
        if os.path.isdir(args.model_dir+str(args.L)):
            shutil.rmtree(args.model_dir+str(args.L))
        os.makedirs(args.model_dir+str(args.L))

    # print the arg pamameters
    for arg in vars(args):
        print('{:<15s}: {:s}'.format(arg,  str(getattr(args, arg))))

    # making traing data
    simulate_dir = Path(args.simulate_dir)
    train_im_list = list(simulate_dir.glob('*.jpg')) + list(simulate_dir.glob('*.png')) + \
    list(simulate_dir.glob('*.tif'))
    train_im_list = sorted([str(x) for x in train_im_list])
    # making tesing data
    test_im_list = (Path('images') / 'val').glob('*.jpg')
    test_im_list = sorted([str(x) for x in test_im_list])
    datasets = {'train':DenoisingDatasets.SimulateTrain(train_im_list, 6000*args.batch_size,
                                          args.patch_size, L=args.L),
                'test_bsd68':DenoisingDatasets.SimulateTest(test_im_list, L=args.L, dep_U =args.depth)}
    # train model
    print('\nBegin training with GPU: ' + str(args.gpu_id))
    train_model(net, datasets, optimizer, scheduler, loss_fn)

if __name__ == '__main__':
    main()

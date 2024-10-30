import argparse

from os import listdir
from os.path import join
import os
import PIL.Image as pil_image
import PIL.ImageFilter as pil_image_filter

import cv2
import numpy as np
import pandas as pd

import torch
from torchvision import transforms

from networks import VDN, weight_init_kaiming
from utils import calc_psnr, calc_ssim, calc_mae, calc_iicc, set_logging, select_device

from tqdm import tqdm

from skimage import img_as_float, img_as_ubyte
import scipy.io as io
use_gpu = True
L = 4
_C = 1
dep_U = 4

def addnoise(Img, L = 4):
    #im_ori = cv2.imread(im_path, 0)#顺序相反操作,变成RGB
    #Img = np.expand_dims(im_ori, 2)
    im_gt = img_as_float(Img)
    # generate noise
    noise_np = np.random.gamma(L, 1/L,im_gt.shape)
    noise = noise_np.astype(np.float32)
    im_noisy = im_gt * noise
    im_noisy = im_noisy.astype(np.float32)
    im_gt = im_gt.astype(np.float32)

    im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1))[np.newaxis,])
    im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1))[np.newaxis,])
    return im_noisy, im_gt


def main() :
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, default = "OURS")
    parser.add_argument("--L", type = str, default = "L10_20")
    parser.add_argument("--weights_dir", type = str, default = "./model10/model_state_20")
    #parser.add_argument("--test_data", type = str, default = r"./data/SAR_test_mat/L1")
    #parser.add_argument("--save_dir", type = str, default = "./our_result1/SARtest")
    parser.add_argument("--test_data", type = str, default = r"./data/MYNWPU_test_mat/L10")
    parser.add_argument("--save_dir", type = str, default = "./our_result1/MYNWPU")
    parser.add_argument("--stack-image", action = "store_true",default = False)
    parser.add_argument("--device", default = "0", help = "cuda device, i.e. 0 or 0,1,2,3 or cpu")
    args = parser.parse_args()
    print(args)

    # Get Current Namespace
    print(args)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # Assign Device
    set_logging()
    device = select_device(args.model_name, args.device)
    print(device)
    # Create Model Instance
    model = VDN(_C, slope=0.2, wf=64, dep_U=4).to(device)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.weights_dir))

    # Create Torchvision Transforms Instance
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Create List Instance for Saving Metrics
    image_name_list, psnr_noisy_list, psnr_denoised_list, ssim_noisy_list, ssim_denoised_list,mae_noisy_list, mae_denoised_list, iicc_noisy_list, iicc_denoised_list = list(), list(), list(), list(), list(), list(), list(), list(), list()

    # Assign Device
    model.to(device)

    # Evaluate Model
    model.eval()

    with tqdm(total = len(listdir(args.test_data))) as pbar :
        psnr_noisy_test = 0
        psnr_denoised_test = 0
        ssim_noisy_test = 0
        ssim_denoised_test = 0
        mae_noisy_test = 0
        mae_denoised_test = 0
        iicc_noisy_test = 0
        iicc_denoised_test = 0
        with torch.no_grad() :

            #read_path = r"./data/sp_noise/p6"
            read_path = args.test_data
            # 获取read_path下的所有文件名称（顺序读取的）
            files = os.listdir(read_path)
            
            files.sort()
            for img in files:
                data = io.loadmat(read_path+"/" + img)

                # ------------------------------------
                # (1) img_L
                # ------------------------------------
                filename, ext = os.path.splitext(os.path.basename(img))
                INoisy = data['f']/255.0
                Io = data['I']/255.0

                clean_image  = np.expand_dims(Io, 0)
                clean_image = np.expand_dims(clean_image, 1)
                noisy_image = np.expand_dims(INoisy, 0)
                noisy_image = np.expand_dims(noisy_image, 1)

                clean_image = torch.Tensor(clean_image)
                noisy_image = torch.Tensor(noisy_image)

                tensor_clean_image = clean_image.cuda()
                tensor_noisy_image = noisy_image.cuda()
                # Get Prediction
                phi_n, pred = model(tensor_noisy_image, 'test')

                # Assign Device into CPU
                tensor_clean_image = tensor_clean_image.detach().cpu()
                tensor_noisy_image = tensor_noisy_image.detach().cpu()
                pred = pred.detach().cpu()


                # Calculate PSNR
                psnr_noisy = calc_psnr(tensor_noisy_image, tensor_clean_image).item()
                psnr_denoised = calc_psnr(pred, tensor_clean_image).item()

                # Calculate SSIM
                ssim_noisy = calc_ssim(tensor_noisy_image, tensor_clean_image,size_average = True).item()
                ssim_denoised = calc_ssim(pred, tensor_clean_image, size_average = True).item()

                # Calculate MAE
                mae_noisy = calc_mae(tensor_noisy_image, tensor_clean_image).item()
                mae_denoised = calc_mae(pred, tensor_clean_image).item()

                # Calculate IICC
                iicc_noisy = calc_iicc(tensor_noisy_image, tensor_clean_image).item()
                iicc_denoised = calc_iicc(pred, tensor_clean_image).item()

                # Append Image Name
                image_name_list.append(filename)

                # Append PSNR
                psnr_noisy_list.append(psnr_noisy)
                psnr_denoised_list.append(psnr_denoised)
                psnr_noisy_test += psnr_noisy
                psnr_denoised_test += psnr_denoised

                # Append SSIM
                ssim_noisy_list.append(ssim_noisy)
                ssim_denoised_list.append(ssim_denoised)
                ssim_noisy_test += ssim_noisy
                ssim_denoised_test += ssim_denoised

                # Append MAE
                mae_noisy_list.append(mae_noisy)
                mae_denoised_list.append(mae_denoised)
                mae_noisy_test += mae_noisy
                mae_denoised_test += mae_denoised

                # Append PSNR
                iicc_noisy_list.append(iicc_noisy)
                iicc_denoised_list.append(iicc_denoised)
                iicc_noisy_test += iicc_noisy
                iicc_denoised_test += iicc_denoised

                # Convert PyTorch Tensor to Pillow Image
                pred = torch.clamp(pred, min = 0.0, max = 1.0)
                pred = to_pil(pred.squeeze(0))

                if args.stack_image :
                    # Get Edge
                    noisy_image_edge = noisy_image.filter(pil_image_filter.FIND_EDGES)
                    pred_edge = pred.filter(pil_image_filter.FIND_EDGES)
                    clean_image_edge = clean_image.filter(pil_image_filter.FIND_EDGES)

                    # Convert into Numpy Array
                    noisy_image = np.array(noisy_image, dtype = "uint8")
                    pred = np.array(pred, dtype = "uint8")
                    clean_image = np.array(clean_image, dtype = "uint8")

                    noisy_image_edge = np.array(noisy_image_edge, dtype = "uint8")
                    pred_edge = np.array(pred_edge, dtype = "uint8")
                    clean_image_edge = np.array(clean_image_edge, dtype = "uint8")

                    # Stack Images
                    stacked_image_clean = np.hstack((noisy_image, pred, clean_image))
                    stacked_image_edge = np.hstack((noisy_image_edge, pred_edge, clean_image_edge))
                    stacked_image = np.vstack((stacked_image_clean, stacked_image_edge))

                    # Save Image
                    cv2.imwrite(f"{args.save_dir}/{filename}", stacked_image)

                else :
                    # Save Image
                    pred.save(f"{args.save_dir}/{filename}_ours.png")

                # Update TQDM Bar
                pbar.update()
            psnr_noisy_test /= len(listdir(args.test_data))
            psnr_denoised_test /= len(listdir(args.test_data))
            ssim_noisy_test /= len(listdir(args.test_data))
            ssim_denoised_test /= len(listdir(args.test_data))

            mae_noisy_test /= len(listdir(args.test_data))
            mae_denoised_test /= len(listdir(args.test_data))
            iicc_noisy_test /= len(listdir(args.test_data))
            iicc_denoised_test /= len(listdir(args.test_data))

            image_name_list.append('Average')
            psnr_noisy_list.append(psnr_noisy_test)
            psnr_denoised_list.append(psnr_denoised_test)
            ssim_noisy_list.append(ssim_noisy_test)
            ssim_denoised_list.append(ssim_denoised_test)

            mae_noisy_list.append(mae_noisy_test)
            mae_denoised_list.append(mae_denoised_test)
            iicc_noisy_list.append(iicc_noisy_test)
            iicc_denoised_list.append(iicc_denoised_test)
            print('psnr_denoised_test', psnr_denoised_test)
            print('ssim_denoised_test', ssim_denoised_test)
            print('mae_denoised_test', mae_denoised_test)
            print('iicc_denoised_test', iicc_denoised_test)

    # Create Dictionary Instance
    d = {"Noisy Image PSNR(dB)" : psnr_noisy_list,
            "Noisy Image SSIM" : ssim_noisy_list,
            "Noisy Image MAE" : mae_noisy_list,
            "Noisy Image IICC" : iicc_noisy_list,
            "Denoised Image PSNR(dB)" : psnr_denoised_list,
            "Denoised Image SSIM" : ssim_denoised_list,
            "Denoised Image MAE" : mae_denoised_list,
            "Denoised Image IICC" : iicc_denoised_list}

    # Create Pandas Dataframe Instance
    df = pd.DataFrame(data = d, index = image_name_list)

    # Save as CSV Format
    df.to_csv(f"{args.save_dir}/our_{args.L}.csv")

if __name__ == "__main__" :
    main()

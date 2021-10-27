import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import os, time, random
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
import cv2
from model.model import InvISPNet
from dataset.holo_dataset import imageDataset_test
from config.config import get_arguments
import matplotlib.pyplot as plt
from utils.commons import denorm, preprocess_test_patch
from tqdm import tqdm
from skimage.measure import compare_psnr,compare_ssim
from glob import glob
import time

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
os.system('rm tmp')


parser = get_arguments()
parser.add_argument("--ckpt", type=str, help="Checkpoint path.") 
parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save results. ")
parser.add_argument("--split_to_patch", dest='split_to_patch', action='store_true', help="Test on patch. ")
args = parser.parse_args()
print("Parsed arguments: {}".format(args))


ckpt_name = args.ckpt.split("/")[-1].split(".")[0]

if args.split_to_patch:
    os.makedirs(args.out_path+"%s/results_metric_%s/"%(args.task, ckpt_name), exist_ok=True)
    out_path = args.out_path+"%s/results_metric_%s/"%(args.task, ckpt_name)
else:
    os.makedirs(args.out_path+"%s/results_%s/"%(args.task, ckpt_name), exist_ok=True)
    out_path = args.out_path+"%s/results_%s/"%(args.task, ckpt_name)

start=time.time()
def main(args):
    # ======================================define the model============================================
    net = InvISPNet(channel_in=3, channel_out=3, block_num=8)
    device = torch.device("cuda:0")
    net.to(device)
    net.eval()
    if os.path.isfile(args.ckpt):
        net.load_state_dict(torch.load(args.ckpt), strict=False)
        print("[INFO] Loaded checkpoint: {}".format(args.ckpt))
    else:
        print("checkpoint is not exist")
        assert 0 
    
    print("[INFO] Start data load and preprocessing")
    RAWDataset = imageDataset_test(opt=args) 
    dataloader = DataLoader(RAWDataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True) 
    print("[INFO] Start test...") 
    psnr=[]
    ssim=[]
    for i_batch, sample_batched in enumerate(tqdm(dataloader)):
        step_time = time.time()
        input_raw, target_phase, target_raw = sample_batched['input_raw'].to(device), sample_batched['target_phase'].to(device), \
                            sample_batched['target_raw'].to(device)
        print(input_raw.shape)
        if args.split_to_patch:
            input_list, target_phase_list, target_raw_list = preprocess_test_patch(input_raw, target_phase, target_raw)
        else:
            # remove [:,:,::2,::2] if you have enough GPU memory to test the full resolution 
            input_list, target_phase_list, target_raw_list = [input_raw], [target_phase], [target_raw]
        for i_patch in range(len(input_list)):
            input_patch = input_list[i_patch]
            target_phase_patch = target_phase_list[i_patch]
            target_raw_patch = target_raw_list[i_patch]
            with torch.no_grad():
                reconstruct_phase = net(input_patch)
                reconstruct_phase = torch.clamp(reconstruct_phase, 0, 1)
            pred_phase= reconstruct_phase.detach().permute(0,2,3,1)
            target_phase_patch = target_phase_patch.permute(0,2,3,1)
            pred_phase = denorm(pred_phase, 255)
            target_phase_patch = denorm(target_phase_patch, 255)
            pred_phase = pred_phase.cpu().numpy().astype(np.float32)
            target_phase_patch = target_phase_patch.cpu().numpy().astype(np.float32)
            psnr_phase=compare_psnr(pred_phase[0,:,:,:],target_phase_patch[0,:,:,:],data_range=255)
            pred_blue=PILImage.fromarray(np.uint8(pred_phase[0,:,:,0]))
            pred_green=PILImage.fromarray(np.uint8(pred_phase[0,:,:,1]))
            pred_red=PILImage.fromarray(np.uint8(pred_phase[0,:,:,2]))
            if os.path.exists(os.path.join(out_path,'blue')):
            	pass
            else:
            	os.mkdir(out_path+"blue")
            pred_blue.save(os.path.join(out_path,'blue',"pred_phase_blue_%04d.png"%i_batch), quality=90, subsampling=1)
            if os.path.exists(os.path.join(out_path,'green')):
            	pass
            else:
            	os.mkdir(out_path+"green")
            pred_green.save(os.path.join(out_path,'green',"pred_phase_green_%04d.png"%i_batch), quality=90, subsampling=1)
            if os.path.exists(os.path.join(out_path,'red')):
            	pass
            else:
            	os.mkdir(out_path+"red")
            pred_red.save(os.path.join(out_path,'red',"pred_phase_red_%04d.png"%i_batch), quality=90, subsampling=1)
            if os.path.exists(os.path.join(out_path,'phs')):
            	pass
            else:
            	os.mkdir(out_path+"phs")
            cv2.imwrite(os.path.join(out_path,"phs","pred_phs_%04d.png"%i_batch),pred_phase[0,:,:,:])

            ###################################test_phase#########################################
            input_phase=pred_phase[0,:,:,:]/255
            input_phase=input_phase.transpose(2,0,1)
            input_phase=torch.tensor(input_phase,dtype=torch.float32)
            input_phase=input_phase.unsqueeze(0).cuda()
            with torch.no_grad():
                reconstruct_img = net(input_phase, rev=True)
            pred_img = reconstruct_img.detach().permute(0,2,3,1)
            pred_img = torch.clamp(pred_img, 0, 1)
            pred_img = denorm(pred_img, 255)
            pred_img = pred_img.cpu().numpy()
            if os.path.exists(os.path.join(out_path,'rec')):
            	pass
            else:
            	os.mkdir(out_path+"rec")
            cv2.imwrite(os.path.join(out_path,'rec',"pred_img_%04d.png"%i_batch),pred_img[0,:,:,:])
            target_raw_patch=target_raw_patch[0,:,:,:].cpu().numpy()
            target_raw_patch=target_raw_patch.transpose(1,2,0)*255
            ###################compute psnr#############################3
            psnr1=compare_psnr(pred_img[0,:,:,:],target_raw_patch,data_range=255)
            ssim1=compare_ssim(pred_img[0,:,:,:],target_raw_patch,multichannel=True,data_range=255)
            print('psnr_phase',psnr_phase,'ssim:',ssim1,'psnr:',psnr1)
            ssim.append(ssim1)
            psnr.append(psnr1)
            del reconstruct_img 
            del reconstruct_phase
    psnr_avg=sum(psnr)/(len(psnr))
    ssim_avg=sum(ssim)/(len(ssim))
    print('psnr_avg',psnr_avg,'ssim_avg',ssim_avg)
    end=time.time()
    print('total time',end-start)

if __name__ == '__main__':
    torch.set_num_threads(4)
    main(args)

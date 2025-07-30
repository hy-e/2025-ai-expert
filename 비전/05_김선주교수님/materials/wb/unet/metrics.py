import os
from pathlib import Path
import math,cv2,torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim

def get_MAE(pred,gt,tensor_type,camera=None,mask=None,pixel_level=False):
    """
    pred : (b,c,w,h)
    gt : (b,c,w,h)
    """
    orig_shape = pred.shape
    if tensor_type == "rgb":
        if camera == 'galaxy':
            pred = torch.clamp(pred, 0, 1023)
            gt = torch.clamp(gt, 0, 1023)
        elif camera == 'sony' or camera == 'nikon':
            pred = torch.clamp(pred, 0, 16383)
            gt = torch.clamp(gt, 0, 16383)

    cos_similarity = F.cosine_similarity(pred+1e-4,gt+1e-4,dim=1)
    cos_similarity = torch.clamp(cos_similarity, -1, 1)
    rad = torch.acos(cos_similarity)
    ang_error = torch.rad2deg(rad) # [B, 256, 256]
    
    if pixel_level:
        return ang_error.detach().cpu()

    if mask is not None:
        ang_error = ang_error[torch.squeeze(mask,1)!=0]
    
    mean_angular_error = ang_error.mean()
    return mean_angular_error

def get_PSNR(pred, gt, white_level):
    """
    pred & gt   : (b,c,h,w) numpy array 3 channel RGB
    returns     : average PSNR of two images
    """
    if white_level != None:
        pred = torch.clamp(pred,0,white_level)
        gt = torch.clamp(gt,0,white_level)

    mse = torch.mean((pred-gt)**2)
    psnr = 20 * torch.log10(white_level / torch.sqrt(mse))

    # pred_np = pred.cpu().numpy()
    # gt_np = gt.cpu().numpy()

    # psnr_cv = cv2.PSNR(pred_np,gt_np,white_level)

    return psnr

def get_SSIM(pred, GT, white_level):
    """
    pred & GT   : (h,w,c) numpy array 3 channel RGB

    returns     : average PSNR of two images
    """
    if white_level != None:
        pred = np.clip(pred, 0, white_level)
        GT = np.clip(GT, 0, white_level)

    return ssim(pred, GT, multichannel=True, data_range=white_level)

def draw_AE_map(ae_map, fname, mae, psnr, ae_map_dir):
    plt.pcolor(ae_map, vmin=0, vmax=60)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(f"file: {fname} / MAE: {mae:.5f} / PSNR: {psnr:.3f}")
    Path(ae_map_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(ae_map_dir, f"{fname}.png"))
    plt.clf()


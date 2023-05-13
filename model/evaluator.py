import torch 
import numpy as np 
from skimage.metrics import structural_similarity

import cv2 
# from skimage.measure import compare_ssim
#### Source: TAVA & Neural Body
# def compute_psnr_tava(pred, target):
#     """Compute psnr value (we assume the maximum pixel value is 1)."""
#     print(f"pred max:{torch.max(pred)}, {torch.min(pred)}")    
#     print(f"target max:{torch.max(target)}, {torch.min(target)}")
#     mse = ((pred - target) ** 2).mean()
#     psnr = -10.0 * torch.log(mse) / np.log(10.0) 
#     return psnr

def compute_psnr(pred, target):
    """Compute psnr value (we assume the maximum pixel value is 1)."""
    # print(f"pred max:{torch.max(pred)}, {torch.min(pred)}")    
    # print(f"target max:{torch.max(target)}, {torch.min(target)}")
    mse = ((pred - target) ** 2).mean()   
    psnr = 20 * np.log10(1) - 10 * torch.log10(mse)
    return psnr


#### TAVA 
def compute_ssim(pred, target):
    """Computes Masked SSIM following the neuralbody paper."""
    assert pred.shape == target.shape and pred.shape[-1] == 3
    if(pred.shape[0]==1):
        pred = torch.squeeze(pred, 0)
        target=torch.squeeze(target, 0)
    # print(f"pred:{torch.max(pred), {torch.min(pred)}}")
    # print(f"target:{torch.max(target), {torch.min(target)}}")
    ssim_val = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), multichannel=True
        )
    ### channel_axis=-1    multichannel=True
    # try:
    #     # ssim = structural_similarity(
    #     #     pred.cpu().numpy(), target.cpu().numpy(), channel_axis=-1
    #     # )
    #     ssim_val = structural_similarity(
    #         pred.cpu().numpy(), target.cpu().numpy(), channel_axis=-1
    #     )
    # except ValueError:
    #     # ssim = structural_similarity(
    #     #     pred.cpu().numpy(), target.cpu().numpy(), multichannel=True
    #     # )
    #     ssim_val = structural_similarity(
    #         pred.cpu().numpy(), target.cpu().numpy(), multichannel=True
    #     )
    
    return ssim_val


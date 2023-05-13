import os 
from os.path import join 
from pathlib import Path 
import matplotlib.pyplot as plt 
import numpy as np
import torch


def plot_gt_vs_predict(gt, pred, fname="313_test.png"):
    if gt.dim()==4: 
        gt = gt[0,:,:,:]
        pred=pred[0,:,:,:]
        
    gt_np = gt.permute(1,2,0).detach().cpu().numpy()
    pred_np = pred.permute(1,2,0).detach().cpu().numpy()
    #### FROM BGR->RGB using numpy
    gt_np = gt_np[...,::-1]
    pred_np=pred_np[...,::-1]
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax1.imshow(gt_np)
    ax1.set_title("gt image")
    ax2.imshow(pred_np)
    ax2.set_title("pred_image")
    save_dir = "/home/user/output/Plots"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fname = "313_test.png"
    save_fpath=join(save_dir,fname)
    plt.savefig(save_fpath, bbox_inches="tight")
    plt.show()
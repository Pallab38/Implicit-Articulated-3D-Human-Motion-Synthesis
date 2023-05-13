import sys
sys.path.insert(0,"")
import os
from os.path import join, isdir, isfile
from pathlib import Path
import numpy as np
from numpy.random import default_rng
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
import cv2
import imageio
import trimesh


# import torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import Callback
from einops import rearrange

from utilities.FourierFeature import input_mapping_torch
from utilities.impCarv_points import PulsarLayer
from utilities.latent_codes import LatentCodes
from utilities.plotting import plot_gt_vs_predict
from dataset_pulsar import PersonPulsar, PersonDatasetPulsar
from network import SimpleMLP
from unet import UNet
from evaluator import compute_ssim, compute_psnr



import warnings
warnings.filterwarnings("ignore")
from typing import Tuple

import argparse
import time 
import logging
import multiprocessing
import smtplib
import ssl 
from email.message import EmailMessage
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("\nPyTorch Lightning Version: ",pl.__version__)
# CUDA_LAUNCH_BLOCKING=1

def send_email(epoch_end=True,epoch_num=0, error=False):
    username = "pallab38@gmail.com"
    password = "bcuxcgbdolibyqvq"
    receiver = "s6padass@uni-bonn.de"

    em = EmailMessage()
    em["From"] = username
    em["To"] = receiver
    em["Subject"] = "Status of Running Processes On the CVG-SRV07 server"   
    if(epoch_end):
        body = f"The epoch {epoch_num} is done. "
    else: 
        body = "The training is finished"
    
    em.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com",465, context=context) as smtp:
        smtp.login(username,password)
        smtp.sendmail(username,receiver, em.as_string())
        
class MyPrintingCallback(Callback):
    def on_train_start(self,trainer,pl_module):
        print("->>>>>>>  Training is starting   <<<<<<<-")
    def on_train_end(self,trainer,pl_module):
        print("->>>>>>>  Training is ending  <<<<<<<-")
        send_email(epoch_end=False)


## https://github.com/Lightning-AI/lightning/issues/2534#issuecomment-674582085
class CheckpointEveryNEpochs(Callback):
    """
    Save a checkpoint every N Epochs
    """
    def __init__(self, save_epoch_frequency, prefix="N_Epoch_Checkpoint",
                 use_modelCheckpoint_filename=False):
        super().__init__()
        self.save_epoch_frequency = save_epoch_frequency
        self.prefix = prefix
        self.use_modelCheckpoint_filename = use_modelCheckpoint_filename
    
    #### https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.Callback.html#pytorch_lightning.callbacks.Callback.on_train_epoch_end
    def on_train_epoch_end(self, trainer, _):
        epoch = trainer.current_epoch
        if epoch % self.save_epoch_frequency==0:
            if self.use_modelCheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename= f"{self.prefix}_{epoch}.ckpt"
            
            dir_path = os.path.dirname(trainer.checkpoint_callback.dirpath)
            save_dir = join(dir_path, "saveEvery_%dEpoch"%self.save_epoch_frequency)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            ckpt_path = join(save_dir, filename)
            trainer.save_checkpoint(ckpt_path)
            
class ShallowNeuralShader(torch.nn.Module):
    def __init__(self, input_channels:int=33, 
                 output_channels:int=4,
                 batch_normalization:bool=False):
        super().__init__()
        self.with_bn = batch_normalization
        self.unet = UNet(in_channels=input_channels, out_channels=output_channels,
                         mid_channels=128, with_bn=self.with_bn)
    def forward(self, neural_image: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        out = self.unet(neural_image)
        img = torch.tanh(out[:,1:4,:,:])
        mask = torch.sigmoid(out[:,0,:,:])
        return img, mask

class PainterDensityNet(torch.nn.Module):
    """
    2 Things Together.
    (i) Painter Net (3dTex, Dir, latent-> bs, n_pts, ch=32)
    (ii) Density Net (3dTex, latent    -> bs, n_pts, ch=1)
    """
    def __init__(self, in_painter: int=1096, in_density:int=584)-> None: 
        super(PainterDensityNet, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.painter = SimpleMLP(input_dim=in_painter, net_type="painter")
        self.density = SimpleMLP(input_dim=in_density, net_type="density")
        
    def forward(self, painter_in, density_in, batch):
        out_painter = self.painter(painter_in) ## [1, 300000, 32]  -0.2069895714521408, 0.40168941020965576
        out_density = self.density(density_in) ## [bs, n_pts, 1] 0.509977400302887, 0.5034168362617493
        return out_painter, out_density


class UnetPainterDensityNet(torch.nn.Module):
    """
    4 Things Together.
    (i) Painter Net (3dTex, Dir, latent-> bs, n_pts, ch=32)
    (ii) Density Net (3dTex, latent    -> bs, n_pts, ch=1)
    (iii) Pulsar Layer ([bs,n_pts,ch=33]-> [bs, h, w, 33])
    (iv) Neural Shader (UNet) [bs, h, w, 33]-> img[bs, ch, h, w]; mask[bs,h,w]
    """
    def __init__(self, in_painter: int=1096, in_density:int=584, 
                 n_pts:int=300000, in_ch:int=33, out_ch:int=4)-> None: 
        super(UnetPainterDensityNet, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.painter = SimpleMLP(input_dim=in_painter, net_type="painter")
        self.density = SimpleMLP(input_dim=in_density, net_type="density")
        self.pulsar = PulsarLayer(n_pts, height=1024, width=1024,
                                  n_channels=in_ch, device=self.device).to(self.device)
        self.neuralShader = ShallowNeuralShader(input_channels= in_ch,output_channels=out_ch)

    def forward(self, painter_in: torch.tensor, density_in:torch.tensor, 
                batch: dict)-> Tuple[torch.tensor, torch.tensor]:
        """
        Args:
            painter_in (torch.tensor): Input for the painter net [bs, n_pts, 1096]
            density_in (torch.tensor): Input for the density net [bs, n_pts, 584]
            batch (dict): containing data

        Returns:
            Tuple[torch.tensor, torch.tensor]: img, mask
        """

        ### (i) Painter Net (3dTex, Dir, latent-> bs, n_pts, ch=32)
        out_painter = self.painter(painter_in) ## [1, 300000, 32]  -0.2069895714521408, 0.40168941020965576
        ### (ii) Density Net (3dTex, latent    -> bs, n_pts, ch=1)
        out_density = self.density(density_in) ## [bs, n_pts, 1] 0.509977400302887, 0.5034168362617493
        # return out_painter, out_density
        ### (iii) Pulsar Layer ([bs,n_pts,ch=33]-> [bs, h, w, 33])
        pulsar_in = torch.cat((out_density, out_painter), dim=2) ## [1, 300000, 33]
        pts_3d, gt_mask  = batch["pts3d"].float(), batch["mask"] ## [bs, n_pts, 3], [bs, ch, h, w]
        vert_radii, rvec = batch["vert_radii"], batch["rvec"] ## [bs, n_pts,1], [bs, 3] 
        Cs, Ks = batch["Cs"], batch["Ks"] ### [bs, 3], [bs, 3, 3]
        pulsar_out = self.pulsar(pts_3d, pulsar_in, vert_radii, rvec, Cs, Ks,opacity=out_density)
        pulsar_out = rearrange(pulsar_out, "b h w c -> b c h w") ## [1, 32, h, w]
        
        ### (iv) Neural Shader (UNet) [bs, h, w, 33]-> img[bs, ch, h, w]; mask[bs,h,w]
        img, mask = self.neuralShader(pulsar_out)
        
        return img, mask
            
        

class LitPainterDensity(pl.LightningModule):
    def __init__(self, my_painterDensity: torch.nn.Module, shallow_neuralSahder: torch.nn.Module):    
        super().__init__()
        self.painterDensity = my_painterDensity
        self.neuralShader = shallow_neuralSahder
        # self.save_hyperparameters()
        self.automatic_optimization= False
        self.metrics = {'mse': torch.tensor(0.0, device=self.device),
                        'psnr': torch.tensor(0.0, device=self.device),
                        'ssim': torch.tensor(0.0, device=self.device),
                        "num_examples": torch.tensor(0.0, device=self.device)
                        }
            
    def forward(self, x : torch.tensor)-> torch.tensor:
        ### It is for inference/prediction
        output = self.painterDensity(x)
        # output = self.neuralShader(output)
        # output = self.my_neuralNet(x)
        return output
    

    def training_epoch_end(self, outputs):
        """
        outputs is a python list containing the batch_dictionary from each batch
        for the given epoch stacked up against each other. 
        """
        epoch = self.current_epoch
        # send_email(epoch_end=True, epoch_num=epoch)
        # print(f"outputs: {type(outputs), outputs}")
        # print(f"outputs[0]: {type(outputs[0]), outputs[0]}")
        # avg_loss = torch.stack([x['loss'] for x in outputs[0]]).mean() ### while using 2 networks
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        ##### using tensorboard logger
        self.logger.experiment.add_scalar("Loss", avg_loss,self.current_epoch)
        epoch_dict = {"loss": avg_loss}
        txt_dir = "/home/user/result/txtFiles2"
        Path(txt_dir).mkdir(parents=True, exist_ok=True)
        painter_txt_fpath = join(txt_dir, "painterWeightsUNet.txt")
        
        """
        # painter_txt =f"{torch.sum(self.painterDensity.painter.my_net[0].weight.data)}"
        # painter_txt += f"    {torch.sum(self.painterDensity.painter.my_net[0].weight.grad.data)}"
        # with open(painter_txt_fpath, "a") as f_p:
        #     f_p.writelines(painter_txt+"\n")
        
       
        # shallow_txt_fpath = join(txt_dir,"shallowWeights.txt")
        # shallow_txt = f"ShallowUNET_down1 weight : {torch.sum(self.neuralShader.unet.down1[0].weight[0])}"
        # shallow_txt += f"    ShallowUNET_down1 grad   : {torch.sum( self.neuralShader.unet.down1[0].weight.grad.data)}"
        # with open(shallow_txt_fpath, "a") as f_sh: 
        #     f_sh.writelines(shallow_txt+"\n")
            
        # unet_txt_fpath = join(txt_dir,"UnetWeights.txt")
        # unet_txt = f"UNET_down1 weight : {torch.sum(self.neuralShader.down1[0].weight[0])}"
        # unet_txt += f"    UNET_down1 grad   : {torch.sum( self.neuralShader.down1[0].weight.grad.data)}"
        # with open(unet_txt_fpath, "a") as f_sh: 
        #     f_sh.writelines(unet_txt+"\n")
        
        """
        # print(f"shallow NN : {self.neuralShader}")
        # print(f"""painter 0 layer weight: 
        #       {torch.sum(self.painterDensity.painter.my_net[0].weight.data),
        #        torch.sum(self.painterDensity.painter.my_net[0].weight.grad.data)}
        #     """)
        # print(f"""ShallowNN weight: 
        #       {torch.sum(self.neuralShader.unet.down1[0].weight[0]),
        #        torch.sum( self.neuralShader.unet.down1[0].weight.grad.data)
        #        }
        #     """) ### torch.sum(self.neuralShader.unet.down1[0].weight[0].grad[0])
        # print(f"shallowNN grad : {self.neuralShader.unet.down1[0].weight.grad.shape,self.neuralShader.unet.down1[0].weight.grad}")
        # for name, params in self.neuralShader.named_parameters(): 
        #     print(name, params.shape)
            
        # exit()
        """
        # print(f"painter 0 layer weight: {torch.sum(self.painterDensity.painter.my_net[0].weight)}")
        # print(f"painter 0 layer grad: {torch.sum(self.painterDensity.painter.my_net[0].weight.grad)}")
        # print(f"density 0 layer weight: {torch.sum(self.painterDensity.density.my_net[0].weight)}")
        # print(f"density 0 layer grad: {torch.sum(self.painterDensity.density.my_net[0].weight.grad)}")
        # print(f"unet down1 layer weight: {torch.sum(self.neuralShader.down1[0].weight)}")
        # print(f"unet down1 layer grad: {torch.sum(self.neuralShader.down1[0].weight.grad)}")
        # print(f"unet mid1 layer weight: {torch.sum(self.neuralShader.mid1[0].weight)}")
        # print(f"unet mid1 layer grad: {torch.sum(self.neuralShader.mid1[0].weight.grad)}")
        # print(f"unet up1 layer weight: {torch.sum(self.neuralShader.up1[0].weight)}")
        # print(f"unet up1 layer grad: {torch.sum(self.neuralShader.up1[0].weight.grad)}")
        """
        

    def get_input_for_network(self, batch, net_type="painter"):
        pts_uv = batch["pts_uv"].float()
        # n_samples = pts_uv.shape[1]
        # rng = default_rng(12345)
        # noise_fixed = rng.uniform(-0.1,0.1,(pts_uv.shape[0], n_samples,1)) ###(2, 300000, 1)
        # uvw = np.concatenate((pts_uv.detach().cpu().numpy(),noise_fixed), axis=2) ###(2, 300000, 3)
        
        uvw = batch["tex3d"]
        gamma_uvw = input_mapping_torch(uvw, self.device) ### [2, 300000, 512]
        # uvw = torch.tensor(uvw,dtype=torch.float32).to(self.device)
        
        latent = batch["latent"] ## [bs1, w512, h512, c3] 
        latent = rearrange(latent, "b w h c -> b c w h") ##[bs, w=512, h=512, ch=3]->[1, 3, 512, 512]
        pts_tex = pts_uv.unsqueeze(2) ## [bs, n_pts, 2]->[bs, n_pts, 1, 2]
        latent_image = torch.nn.functional.grid_sample(latent, pts_tex, 
                                                     align_corners=True) ### (b=1, c=3, w=n_pts=300000, h=1)
        latent_image = rearrange(latent_image,"b c w h -> b (w h) c")    
        # print(f"latent_image.shape:{latent_image.shape}, latent_dim:{latent_image.shape[2]}")
        self.my_latent_dim = latent_image.shape[2]
        if(self.my_latent_dim==16):
            latent_image=latent_image[:,:,:3]
            
        fft_normDirection = batch["fft_normDirection"]
        smpl_poses = batch["smpl_poses"] ## [1, 1, 72]
        smpl = smpl_poses.repeat(1, latent_image.shape[1], 1) ### [1, 300000, 72] 
        ### smpl.shape[2]=72/3 == 24 
        lt_img = torch.repeat_interleave(latent_image, 24, dim=2)### [1, 300000, 72] 
        latent_smpl = lt_img + smpl ### [1, 300000, 72]
        
        if net_type=="painter":
            #### Concat  gamma_pts= [1, 300000, 512]  latent_smpl= [1, 300000, 72]
            my_input = torch.cat((gamma_uvw, latent_smpl,fft_normDirection), dim=2).to(torch.float32)  ## [1, 300000, 1096]
            my_input = torch.tensor(my_input).to(self.device)
        else: 
            my_input = torch.cat((gamma_uvw, latent_smpl), dim=2).to(torch.float32)
            my_input = torch.tensor(my_input).to(self.device)

        return my_input

    def save_all_images(self, my_epoch:int, pid:str, view_cam:int,
                        shader_img:torch.tensor,shader_pred_img:torch.tensor,
                        pulsar_img:torch.tensor, pred_img:torch.tensor):
        """Saves 4 images

        Args:
            my_epoch (int): epoch number
            bn (str): withbn or without_bn
            pid (str): 377 or 313
            shader_img (torch.tensor): RGB image from neural shader
            shader_pred_img (torch.tensor): gt_mask * RGB image from neural shader
            pulsar_img (torch.tensor): RGB image from pulsar
            pred_img (torch.tensor): gt_mask * RGB image from pulsar
        """
        # print(f"shader_img:{torch.min(shader_img).item(), torch.max(shader_img).item()}")
        # print(f"shader_pred_img:{torch.min(shader_pred_img).item(), torch.max(shader_pred_img).item()}")
        # print(f"pulsar_img:{torch.min(pulsar_img).item(), torch.max(pulsar_img).item()}")
        # print(f"pulsar_pred_img:{torch.min(pred_img).item(), torch.max(pred_img).item()}")
        
        ### Change image value range [-1,1] to [0,1]
        shader_img = (shader_img[0,:,:,:]+1)/2
        shader_pred_img = (shader_pred_img[0,:,:,:]+1)/2
        pulsar_img = (pulsar_img[0,:,:,:]+1)/2
        pred_img = (pred_img[0,:,:,:]+1)/2
        
        save_root=join("/home/user/output/NeuralShader",f"{pid}_AllFrames")
        save_cnn=join(save_root,"cnn_img")
        save_cnnXmask=join(save_root,"cnn_imgXmask")
        save_pulsar=join(save_root,"pulsar_img")
        save_pulsarXmask=join(save_root,"pulsar_imgXmask")
        Path(save_root).mkdir(parents=True,exist_ok=True)
        Path(save_cnn).mkdir(parents=True,exist_ok=True)
        Path(save_cnnXmask).mkdir(parents=True,exist_ok=True)
        Path(save_pulsar).mkdir(parents=True, exist_ok=True)
        Path(save_pulsarXmask).mkdir(parents=True, exist_ok=True)
        
        cnn_fpath= join(save_cnn,f"epoch_{my_epoch}_C{view_cam}.png")
        cnnXmask_fpath = join(save_cnnXmask,f"epoch_{my_epoch}_C{view_cam}.png")
        pulsar_fpath = join(save_pulsar, f"epoch_{my_epoch}_C{view_cam}.png")
        pulsarXmask_fpath=join(save_pulsarXmask,f"epoch_{my_epoch}_C{view_cam}.png")
        save_image(shader_img,cnn_fpath)
        save_image(shader_pred_img, cnnXmask_fpath)
        save_image(pulsar_img, pulsar_fpath)
        save_image(pred_img, pulsarXmask_fpath)
        
    
    def training_step(self, batch, batch_idx):
        # print(" ")                
        pid, gt_img, gt_mask = batch["pid"], batch["image"], batch["mask"] ## "313",(1, 3, 512, 512),(1, 1, 512, 512)
        painter_input = self.get_input_for_network(batch, net_type="painter") ### [bs,n_pts,1096],
        density_input = self.get_input_for_network(batch, net_type="density") ### [bs,n_pts,584],
        
        out_painter, out_density = self.painterDensity(painter_input, density_input, batch) 
        ## [bs,n_pts,32],[-1,1]   [bs, n_pts,1],[0,1]
        # print(f"out_painter: {out_painter.shape, torch.min(out_painter).item(), torch.max(out_painter).item()}")
        # print(f"out_density: {out_density.shape, torch.min(out_density).item(), torch.max(out_density).item()}")
           
        pulsar_in = torch.cat((out_density, out_painter), dim=2) ## [1, 300000, 33]
        # print(f"pulsar_in:{pulsar_in.shape, torch.min(pulsar_in).item(), torch.max(pulsar_in).item()}")
        pts_3d, gt_mask  = batch["pts3d"].float(), batch["mask"] ## [bs, n_pts, 3], [bs, ch, h, w]
        vert_radii, rvec = batch["vert_radii"], batch["rvec"] ## [bs, n_pts,1], [bs, 3] 
        Cs, Ks = batch["Cs"], batch["Ks"] ### [bs, 3], [bs, 3, 3]
        pulsar = PulsarLayer(n_pts, height=1024, width=1024,
                            n_channels=33, device=self.device).to(self.device)
        ###[bs1, n_pts300000, ch1] -> [1, 300000]
        out_dense = rearrange(out_density,"b h c -> b (h c)")
        # print(f"out_density: {out_dense.shape, torch.sum(out_dense)}")
        ### opacity: [Bx]N tensor of opacity values in [0., 1.] or None (uses all ones).
        pulsar_out = pulsar(pts_3d, pulsar_in, vert_radii, rvec, 
                            Cs, Ks, opacity=out_dense) ### [bs, h=1024, w=1024, ch=33]
        # print(f"pulsar_out: {pulsar_out.shape, torch.min(pulsar_out).item(), torch.max(pulsar_out).item()}")
        pulsar_out = rearrange(pulsar_out, "b h w c -> b c h w") ## [1, 33, h, w]
        pulsar_mask = pulsar_out[:, 0, :, :]  ### [1, 1024, 1024]
        pulsar_img = pulsar_out[:, 1:4,:, :]  ### [1, 3, 1024, 1024] (-0.6411585807800293, 0.0) w/ density
        pulsar_pred_image = pulsar_img * gt_mask
        
        # print(f"NeuralShader Input shape:{pulsar_out.shape} ")
        shader_img,  shader_mask = self.neuralShader(pulsar_out)
        # print(f"shader_img: {shader_img.shape, torch.min(shader_img).item(), torch.max(shader_img).item()}")
        # print(f"shader_mask: {shader_mask.shape, torch.min(shader_mask).item(), torch.max(shader_mask).item()}")
        shader_pred_img = shader_img * gt_mask
                
        
        epoch = trainer.current_epoch
        
        #"Since 1 mesh and All Cam"
        subject_view_dict={"313":1, "315":1,"377":5,"386":8, "387":5,
                           "390":5, "392":5, "393":7, "394":3}
        person = pid[0]
        view_cam = subject_view_dict[person]        
        if epoch%10==0 and batch_idx==view_cam-1:
            self.save_all_images(epoch, person, view_cam, 
                                 shader_img, shader_pred_img, 
                                 pulsar_img, pulsar_pred_image)            
            
        
        #### gt_image = gt_img * gt_mask     ## [1, 3, 1024, 1024],torch.float32, 0.0 , 0.0039 
        ### [0,1] to [-1,1] x_norm = 2 * (x-torch.min(x))/ (torch.max(x)-torch.min(x)) -1
        gt_norm = 2 * (gt_img - torch.min(gt_img)) / (torch.max(gt_img)- torch.min(gt_img))-1  ### (-1.0, 1.0)
        gt_image = gt_norm * gt_mask     ## [1, 3, 1024, 1024],torch.float32, 0.0 , 0.0039 (-1.0, 1.0)
        l1_loss = nn.L1Loss()
        # print(f"gt_image: {gt_image.shape, torch.min(gt_image).item(), torch.max(gt_image).item()}")
        loss_pulsar_img = l1_loss(gt_image, pulsar_pred_image)
        # loss_img = l1_loss(gt_image, unet_img) ###  (-0.08053641021251678, 1.0), (0.0,1.0)
        loss_img=l1_loss(gt_image, shader_pred_img)
        
        pulsar_mask = pulsar_mask.unsqueeze(1) ## [1, 1024, 1024]-> [1, 1, 1024, 1024]
        pred_mask = (pulsar_mask-torch.min(pulsar_mask)) / (torch.max(pulsar_mask)- torch.min(pulsar_mask)) ### [1, 1, 1024, 1024] [0,1] 0.0, 1.0
        # print(f"pred_image value: {torch.min(pred_image).item(), torch.max(pred_image).item()}") ## (0.0, 0.38931334018707275)
        # unet_mask = unet_mask.unsqueeze(1)
        # unet_mask = (unet_mask-torch.min(unet_mask)) / (torch.max(unet_mask)- torch.min(unet_mask)) ### [bs, ch1, h ,w] To make sure the value between[0,1]
        
        bce_loss = nn.BCELoss()        
        loss_pulsar_mask = bce_loss(pred_mask, gt_mask)
        gt_mask = gt_mask.squeeze(1) ## [bs, ch, h, w] => [bs, h, w]
        # loss_mask = bce_loss(unet_mask, gt_mask) ### (0.0, 1.0), (0.0, 1.0)
        loss_mask = bce_loss(shader_mask, gt_mask)
        
        # total_loss = loss_pulsar_img + loss_pulsar_mask
        total_loss = loss_pulsar_img + loss_img + loss_mask + loss_pulsar_mask
        self.log("total_loss", total_loss, rank_zero_only=True)
        out_dict = {"loss": total_loss, "epoch" : trainer.current_epoch}
        # print(f"total_loss: {total_loss}")
        optim = self.optimizers()
        optim.zero_grad()
        self.manual_backward(total_loss)
        optim.step()
        return out_dict
    
    
    def test_epoch_end(self, outputs):
        print("TEST IS DONE")
        print(f"outputs: {type(outputs), len(outputs)}")
        print(outputs[0])
        metric_dict = outputs[0]
        # print(f"mse: {metric_dict['mse']}")
        # print(f"psnr:{metric_dict['psnr']}")
        # print(f"ssim:{metric_dict['ssim']}") 
        # print(f"num_examples:{metric_dict['num_examples']}")
        
        num_ex=metric_dict["num_examples"]
        mse = metric_dict['mse']/num_ex
        psnr = metric_dict["psnr"]/num_ex
        ssim = metric_dict["ssim"]/num_ex
        metric_txt=join("/home/user/person_313/NeuralShader/painterDensity/TestMetrics/SummaryMetrics")
        with open(metric_txt,"a") as f_metric:
            f_metric.writelines("mse: "+str(mse)+"\n")
            f_metric.writelines("psnr: "+str(psnr)+"\n")
            f_metric.writelines("ssim: "+str(ssim)+"\n")
        print("Average Metrics Values over ALL Test Examples")
        print(f"mse : {mse}")
        print(f"psnr:{psnr}")
        print(f"ssim:{ssim}")
        # exit()
        # for key, value in metric_dict.items():
        #     metric_dict[key] = value / len(index_list_all)
    
    def save_pulsar_img(self, pulsar_img, batch_num, cid):
        save_rootDir= "/home/user/ByCamResults"
        pulsarSaveDir=join(save_rootDir,f"Pulsar_ImXmask/cam{cid}")
        Path(pulsarSaveDir).mkdir(parents=True, exist_ok=True)
        pulsarXmask_fpath=join(pulsarSaveDir,"%06d.png"%batch_num)
        save_image(pulsar_img[0,:,:,:],pulsarXmask_fpath) 
        
    def save_cnn_img(self, cnn_img, batch_num, cid):
        save_rootDir= "/home/user/ByCamResults"
        mySaveDir=join(save_rootDir,f"Cnn_ImXmask/cam{cid}")
        Path(mySaveDir).mkdir(parents=True, exist_ok=True)
        cnnXmask_fpath=join(mySaveDir,"%06d.png"%batch_num)
        save_image(cnn_img[0,:,:,:],cnnXmask_fpath)  
        
    
    def test_step(self, batch, batch_idx):
        pid, gt_img, gt_mask = batch["pid"], batch["image"], batch["mask"] ## "313",(1, 3, 512, 512),(1, 1, 512, 512)
        painter_input = self.get_input_for_network(batch, net_type="painter") ### [bs,n_pts,1096],
        density_input = self.get_input_for_network(batch, net_type="density") ### [bs,n_pts,584],
        
        out_painter, out_density = self.painterDensity(painter_input, density_input, batch) 
        ## [bs,n_pts,32],[-1,1]   [bs, n_pts,1],[0,1]
        pulsar_in = torch.cat((out_density, out_painter), dim=2) ## [1, 300000, 33]
        # print(f"pulsar_in:{pulsar_in.shape, torch.min(pulsar_in).item(), torch.max(pulsar_in).item()}")
        pts_3d, gt_mask  = batch["pts3d"].float(), batch["mask"] ## [bs, n_pts, 3], [bs, ch, h, w]
        vert_radii, rvec = batch["vert_radii"], batch["rvec"] ## [bs, n_pts,1], [bs, 3] 
        Cs, Ks = batch["Cs"], batch["Ks"] ### [bs, 3], [bs, 3, 3]
        pulsar = PulsarLayer(n_pts, height=1024, width=1024,
                            n_channels=33, device=self.device).to(self.device)
        ###[bs1, n_pts300000, ch1] -> [1, 300000]
        out_dense = rearrange(out_density,"b h c -> b (h c)")
        # print(f"out_density: {out_dense.shape, torch.sum(out_dense)}")
        ### opacity: [Bx]N tensor of opacity values in [0., 1.] or None (uses all ones).
        pulsar_out = pulsar(pts_3d, pulsar_in, vert_radii, rvec, 
                            Cs, Ks, opacity=out_dense) ### [bs, h=1024, w=1024, ch=33]
        # print(f"pulsar_out: {pulsar_out.shape, torch.min(pulsar_out).item(), torch.max(pulsar_out).item()}")
        pulsar_out = rearrange(pulsar_out, "b h w c -> b c h w") ## [1, 33, h, w]
        pulsar_mask = pulsar_out[:, 0, :, :]  ### [1, 1024, 1024]
        pulsar_img = pulsar_out[:, 1:4,:, :]  ### [1, 3, 1024, 1024] (-0.6411585807800293, 0.0) w/ density
        pulsar_pred_image = pulsar_img * gt_mask
        pulsar_mask = pulsar_mask.unsqueeze(1) ## [1, 1024, 1024]-> [1, 1, 1024, 1024]
        pred_mask = (pulsar_mask-torch.min(pulsar_mask)) / (torch.max(pulsar_mask)- torch.min(pulsar_mask)) ### [1, 1, 1024, 1024] [0,1] 0.0, 1.0

        # print(f"NeuralShader Input shape:{pulsar_out.shape} ")
        shader_img,  shader_mask = self.neuralShader(pulsar_out)
        # print(f"shader_image:{shader_img.shape}, shader_mask:{shader_mask.shape}")
        # print(f"shader_img: {shader_img.shape, torch.min(shader_img).item(), torch.max(shader_img).item()}")
        # print(f"shader_mask: {shader_mask.shape, torch.min(shader_mask).item(), torch.max(shader_mask).item()}")
        ### [-1,1] to [0,1]
        shader_pred_img = (shader_img[:,:,:,:]+1)/2
        # print(f"shader_img: {torch.min(shader_pred_img).item(), torch.max(shader_pred_img).item()}")
        shader_pred_img = shader_pred_img * gt_mask
        # shader_pred_img = shader_img * gt_mask
        # print(f"shader_img: {torch.min(shader_pred_img).item(), torch.max(shader_pred_img).item()}")
                
        gt_image = gt_img * gt_mask     ## [1, 3, 1024, 1024],torch.float32, 0.0 , 0.0039 
        ### [0,1] to [-1,1] x_norm = 2 * (x-torch.min(x))/ (torch.max(x)-torch.min(x)) -1
        # gt_norm = 2 * (gt_img - torch.min(gt_img)) / (torch.max(gt_img)- torch.min(gt_img))-1  ### (-1.0, 1.0)
        # gt_image = gt_norm * gt_mask     ## [1, 3, 1024, 1024],torch.float32, 0.0 , 0.0039 (-1.0, 1.0)
        # print(f"gt_image: {gt_image.shape, torch.min(gt_image).item(), torch.max(gt_image).item()}")
        
        # ### [-1,1] to [0,1]
        # shader_pred_img = (shader_pred_img[:,:,:,:]+1)/2
        # print(f"shader_img: {torch.min(shader_pred_img).item(), torch.max(shader_pred_img).item()}")
        mse = ((shader_pred_img - gt_image) ** 2).mean()
        psnr=compute_psnr(shader_pred_img, gt_image)
        # print(f"psnr value: {psnr}")
        
        # plot_gt_vs_predict(gt_image, shader_pred_img)## img shapes:bwhc

        shader_pred_img= rearrange(shader_pred_img,"b c w h -> b w h c")
        gt_image = rearrange(gt_image, "b c w h -> b w h c")
        ssim_val = compute_ssim(shader_pred_img, gt_image)
        
        testMetric_dir="/home/user/person_313/NeuralShader/painterDensity/TestMetrics"
        Path(testMetric_dir).mkdir(parents=True, exist_ok=True)
        assert isdir(testMetric_dir), "!!  Directory for test metrics do not exist  !!"
        mse_fpath = join(testMetric_dir,"mse.txt")
        psnr_fpath = join(testMetric_dir,"psnr.txt")
        ssim_fpath = join(testMetric_dir,"ssim.txt")
        with open(mse_fpath,"a") as f_mse:
            f_mse.writelines(str(mse.item())+"\n")
        with open(psnr_fpath,"a") as f_psnr:
            f_psnr.writelines(str(psnr.item())+"\n")
        with open(ssim_fpath,"a") as f_ssim:
            f_ssim.writelines(str(ssim_val.item())+"\n")
        
        
        # print(f"mse: {mse.item()}")
        # print(f"psnr: {psnr.item()}")
        # print(f"ssim:{ssim_val.item()}")
        num_examples= batch["image"].shape[0]
        self.metrics["num_examples"]+=num_examples
        self.metrics["mse"] += mse.detach().cpu()
        self.metrics["psnr"] += psnr.detach().cpu()
        # self.metrics["ssim"] += ssim_val.detach().cpu().numpy()
        self.metrics["ssim"] += torch.from_numpy(np.asarray(ssim_val))
        
        
        
        return self.metrics
         

    def configure_optimizers(self):
        # optimizer1 = optim.Adam(self.painterDensity.parameters(),lr=0.001, betas=(0.9, 0.999))
        # optimizer2 = optim.Adam(self.neuralShader.parameters(), lr=0.001, betas=(0.9,0.999) )
        # return optimizer1, optimizer2
        ### https://pytorch.org/docs/stable/optim.html#per-parameter-options
        optimizer = optim.Adam([{'params': self.painterDensity.parameters()}, 
                                {'params': self.neuralShader.parameters()}],
                               lr=0.001, betas=(0.9, 0.999))
        
        return optimizer
        


class ZJUDataModule(pl.LightningDataModule):
    def __init__(self, root_dir:str, person_ids:list, 
                 sampling_points:int=300000, batch_size:int=4,
                 dataset_type:str="training",
                 latent_path:str="/home"):
        super().__init__()
        self.prepare_data_per_node = True
        self.root_dir = root_dir
        self.person_ids= person_ids
        self.n_pts = sampling_points
        self.bs = batch_size
        self.dataset_type = dataset_type
        self.latent_path = latent_path
    
    def setup(self, stage:str=None):
        #### stage: "fit", "validate", "test", "predict".
        if stage in (None,"fit"):
            Persons =[]
            Datasets = []
            trans = T.Compose([T.ToTensor(), T.ToPILImage(), 
                               T.Resize(1024),T.ToTensor()])
            # trans = T.Compose([T.ToTensor()])
            for person_id in self.person_ids:
                person = PersonPulsar(person_id, root= self.root_dir,
                                      dataset_type=self.dataset_type)
                person_dataset = PersonDatasetPulsar(person, sampling_points=self.n_pts, 
                                                     transform=trans, 
                                                     latent_path= self.latent_path)
                print(f"latent_dir Path:{self.latent_path}")
                print("person: ", person_id, "dataset length: ", len(person_dataset))
                Datasets.append(person_dataset)
                Persons.append(person)
            self.data_set = ConcatDataset(Datasets)  
                      
        else: 
            ### test dataset ###
            Persons =[]
            Datasets = []
            trans = T.Compose([T.ToTensor(), T.ToPILImage(), T.Resize(1024),T.ToTensor()])
            # trans = T.Compose([T.ToTensor()])
            print("Create Dataset for test")
            for person_id in self.person_ids:
                person = PersonPulsar(person_id, root= self.root_dir,
                                      dataset_type=self.dataset_type)
                person_dataset = PersonDatasetPulsar(person, sampling_points=self.n_pts, 
                                                     transform=trans, 
                                                     latent_path= self.latent_path)
                print(f"latent_dir Path:{self.latent_path}")
                print("person: ", person_id, "dataset length: ", len(person_dataset))
                Datasets.append(person_dataset)
                Persons.append(person)
            self.data_set = ConcatDataset(Datasets)  
        
        
    def train_dataloader(self):
        zju_train = DataLoader(self.data_set, batch_size=self.bs, 
                                shuffle=False, num_workers=0,
                                pin_memory=False, drop_last=True)#
        total_frames = len(self.data_set)
        print("******** Train DataLoader ********")
        print("Total : {0}, batch: {1}, batch_size:{2}".format(total_frames, len(zju_train), self.bs))
        return zju_train
    
    def test_dataloader(self):
        zju_test = DataLoader(self.data_set, batch_size=self.bs, 
                                shuffle=False, num_workers=0,
                                pin_memory=False, drop_last=True)#
        total_frames = len(self.data_set)
        print("******** Test DataLoader ********")
        print("Total : {0}, batch: {1}, batch_size:{2}".format(total_frames, len(zju_test), self.bs))
        return zju_test
    
    
    
if __name__=="__main__": 
    pl.utilities.seed.seed_everything(1310)
    print(f"\n nGPUs per node: {torch.cuda.device_count()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid",type=int, default=0, help="Pid:0 for 313 or 2 for 377")
    parser.add_argument("--bs", type=int, default=1, action="store", help="Batch Size")
    parser.add_argument("--run_type", type=str, default="training", help="training or testing")
    parser.add_argument("--trainedModel", action="store_true", help="Training From Trained Model")
    parser.add_argument("--newModel", dest="trainedModel", action="store_false")
    parser.set_defaults(trainedModel=False)
    parser.add_argument("--resume_training", action="store_true", help="Resume Training from already trained .ckpt file")
    parser.set_defaults(resume_training=False)
    args = parser.parse_args() ### Namespace(bs=1, run_type='training', trainedModel=False)
    print(f"arguments from command line: \n{args}")
    run_type=args.run_type
    
    load_checkpoint = args.trainedModel if run_type=="training" else True  # False #True
    bs = args.bs
    print(f"run_type: {run_type}, load_checkpoint:{load_checkpoint}")
    
    pid_dict={0:"313",1:"315",2:"377",3:"386",4:"387",5:"390",6:"392",7:"393",8:"394"}
    person_idx=args.pid
    person=pid_dict[person_idx]
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"!!!!----- {person} ALL FRAMES ----!!!!!!")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    output_dirpath="/home/user/output/Logger"
    Path(output_dirpath).mkdir(parents=True, exist_ok=True)

    my_logger = logging.getLogger(__name__)
    my_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(join(output_dirpath, f"{run_type}NeuralShader_{person}.txt"))
    fh.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s - %(name)s  - %(levelname)s - %(message)s " )
    fh.setFormatter(formatter)
    my_logger.addHandler(fh)
    
    try:
        root_dir = "/home/user/my_zjuMocap"
        out_dir = f"/home/user/my_model/NeuralShader/Person_{person}" 
        pl_log_dir = join(out_dir, "FullModel") ### running_logs
        Path(pl_log_dir).mkdir(parents=True,exist_ok=True)
        assert isdir(pl_log_dir)==True, "pl_log_dir is not created" 
        print(f"model saving path for neural shader person {person}:{pl_log_dir}")  
        training_path= pl_log_dir
        logger = TensorBoardLogger(pl_log_dir, name="pl_model")
        pretrain_dir = "/home/user/LearnTexturePoints_and_LatentCodes"
        ### Latent Codes should be from `SinglePose` even for Multipose training & testing
        latent_path=join(pretrain_dir,f"Person_{person}/SinglePose/AllCamLatent") 
        ##/home/user/DATA/LearnTexturePoints_and_LatentCodes/Person_313/SinglePose/AllCamLatent
                
        n_pts=300000
        zju_trainLoader = ZJUDataModule(root_dir=root_dir, person_ids=[person_idx],
                                        sampling_points=300000, batch_size=bs,
                                        dataset_type=run_type, latent_path=latent_path)
        
        ####==============================================================####        
        #####   PAINTER   NET   #####
        latent_dim = 16  ## 16
        enc_dim_x = 512 ### After 3D pts/ 3D texture pts encoding with Fourier Feature Mapping
        enc_dim_d = 512 ### View Direction encoded with Fourier Feature Mapping
        latent_smpl_dim = 72 ### latent_image + smpl_poses
        in_dim_painter = enc_dim_x + latent_smpl_dim + enc_dim_d ## 1096
        in_dim_density = enc_dim_x + latent_smpl_dim
        ####==============================================================####        
        
        ################################################
        ######    Painter   And  Density  Network  ####
        ################################################        
        ## 1.AllCamLatent TrainCam Model: /home/das1/DATA/my_model/NeuralShader/Person_313_16Ch/AllCamLatent/SinglePose/pl_model/version_0/checkpoints
        # model_path=join(pl_log_dir,"pl_model","version_0","checkpoints","epoch=10-step=29645.ckpt") ## First 10 steps
        # model_path=join(pl_log_dir,"pl_model","version_1","checkpoints","epoch=49-step=187283.ckpt")  ## Then till 50
        # model_path=join(pl_log_dir,"pl_model","version_4","checkpoints","epoch=99-step=389383.ckpt")  ## Then till 100
        model_path=join(pl_log_dir,"pl_model","version_5","checkpoints","epoch=199-step=793583.ckpt")  ## Then till 100
        print(f"model_path:{model_path}")
        if (load_checkpoint==True or run_type=="testing"):
            assert isfile(model_path)==True, "Trained Model Not Found"
        
        painter_density_net = PainterDensityNet(in_painter= in_dim_painter, 
                                                in_density= in_dim_density)
        input_ch, output_ch = 33, 4 
        my_neuralShader  = ShallowNeuralShader(input_channels=input_ch,
                                               output_channels=output_ch
                                              )
        
        ###================================================================================================###
        if load_checkpoint: 
            print("\n ========  Resuming From CheckPoint  ========")            
            pl_net = LitPainterDensity.load_from_checkpoint(
                checkpoint_path=model_path, my_painterDensity=painter_density_net, 
                shallow_neuralSahder= my_neuralShader)            
        else:
            print("\n========Running Training From Scratch========")            
            pl_net = LitPainterDensity(painter_density_net, my_neuralShader)
            
        ## ============================================================================#####
        ### Trainer Class API https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#methods
        ### strategy="ddp_find_unused_parameters_false", "horovod"
        saveCkptPerEpoch = CheckpointEveryNEpochs(5)
        if(args.run_type=="training"): 
            if args.resume_training:
                print("Resuming From Checkpoint (PL argument)")
                #### devices=1, accelerator="gpu"
                trainer = pl.Trainer(gpus=2, accelerator="gpu",strategy="ddp_find_unused_parameters_false",  
                                     limit_train_batches=1.0, max_epochs=200, 
                                     default_root_dir=training_path,
                                     logger=logger, resume_from_checkpoint=model_path,
                                     callbacks=[MyPrintingCallback(),
                                                saveCkptPerEpoch])
            else:
                print("Normal Training")
                
                ### #### devices=1, accelerator="gpu"
                trainer = pl.Trainer(gpus=2, accelerator="gpu",strategy="ddp_find_unused_parameters_false",  
                                     limit_train_batches=1.0, max_epochs=50, 
                                     default_root_dir=training_path,
                                     logger=logger,
                                     callbacks=[MyPrintingCallback(),
                                                saveCkptPerEpoch])
            trainer.fit(pl_net, zju_trainLoader)#, zju_valLoader)
            
        elif(args.run_type=="testing"):
            #########    TEST    (Novel-View Synthesis)   #######   
            print("===== TESTING  =====")     
            trainer = pl.Trainer(devices=1, accelerator="gpu", 
                                limit_train_batches=0, max_epochs=1,  
                                default_root_dir=training_path,
                                logger=logger,
                                callbacks=[MyPrintingCallback()])
            
            trainer.test(pl_net, zju_trainLoader)
        else:
            print(f"Invalid args.run_type= {args.run_type}, accepted arguments 'training' or 'testing' ")

    except Exception as e: 
        my_logger.exception(e)
        time.sleep(3)
        print(e)
        print(type(e))
        # email_for_error(body=str(e))


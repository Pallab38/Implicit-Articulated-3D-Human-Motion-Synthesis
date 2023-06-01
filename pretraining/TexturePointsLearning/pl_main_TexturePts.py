import sys
sys.path.insert(0,"")
import os
from os.path import join
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

from utilities.FourierFeature import input_mapping, input_mapping_torch
from utilities.impCarv_points import PulsarLayer
from utilities.latent_codes import LatentCodes
from utilities.texturePoints import TexturePoints
from dataset_pulsar import PersonPulsar, PersonDatasetPulsar
from network import SimpleMLP

import warnings
warnings.filterwarnings("ignore")
import argparse
import time 
import logging
import multiprocessing
import smtplib
import ssl 
from email.message import EmailMessage
from typing import Tuple, List
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("\nPyTorch Lightning Version: ",pl.__version__)
# CUDA_LAUNCH_BLOCKING=1

class MyPrintingCallback(Callback):
    def on_train_start(self,trainer,pl_module):
        print("->>>>>>>  Training is starting   <<<<<<<-")
    def on_train_end(self,trainer,pl_module):
        print("->>>>>>>  Training is ending  <<<<<<<-")


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
            

class PainterDensityNet(torch.nn.Module):
    """Painter Density Network
    2 Things Together.
    (i) Painter Net (3dTex, Dir, latent-> bs, n_pts, ch=32)
    (ii) Density Net (3dTex, latent    -> bs, n_pts, ch=1)    
    """
    def __init__(self, in_painter: int=1096, in_density:int=584)-> None: 
        """Create Painter Density Network

        Args:
            in_painter (int, optional): Input Dimension for Painter Network. Defaults to 1096.
            in_density (int, optional): Input Dimension for Density Network. Defaults to 584.
        """
        super(PainterDensityNet, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.painter = SimpleMLP(input_dim=in_painter, net_type="painter")
        self.density = SimpleMLP(input_dim=in_density, net_type="density")
        
    def forward(self, painter_in:torch.Tensor, 
                density_in: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """Forward  Function for Painter Density Network.

        Args:
            painter_in (torch.Tensor): Input For The Painter Network.
            density_in (torch.Tensor): Input For the Density Network.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output {bs, n_pts, 32} of the Painter Network, 
                                               Output {bs, n_pts, 1} of the density Network
        """
        
        out_painter = self.painter(painter_in) ## [1, 300000, 32]  -0.2069895714521408, 0.40168941020965576
        out_density = self.density(density_in) ## [bs, n_pts, 1] 0.509977400302887, 0.5034168362617493
        return out_painter, out_density
    

class LitPainterDensity(pl.LightningModule):
    def __init__(self, my_painterDensity: torch.nn.Module,
                person:int, mesh_num:int, n_sample:int, 
                save_path:str, init_epoch:int=-1, my_device: torch.device="cuda"):
        """PyTorch Lightning Painter Density Network With 3D Texture Points Learning.

        Args:
            my_painterDensity (torch.nn.Module): Painter Density Network.
            pid (List[int]): Id of the Person in List: [313].
            mesh_num (int): Mesh Number. Only one mesh: 000000.obj
            n_sample (int): Number of sampling points: 300k.
            save_path (str): Learned 3d Texture Points Path. 
            init_epoch (int, optional): Initial Epoch for 3D Texture Points Learning.
                                        Defaults to -1.
            my_device (torch.device, optional): Device Type. Defaults to "cuda".
        """
        
        super().__init__()
        torch.manual_seed(1310)
        #### Activate MANUAL OPTIMIZATION ####
        self.automatic_optimization = False

        self.person_list = [person]
        self.init_epoch = init_epoch
        self.painterDensity = my_painterDensity
        self.my_texPts = TexturePoints(self.person_list, mesh_num, n_sample,
                                    self.init_epoch, save_path, my_device)
        self.tex_pts_dict = self.my_texPts.data[self.person_list[0]]
        self.points_texture = self.tex_pts_dict["tex_pts"]
        self.tex_uv = self.tex_pts_dict["tex_2d"]
        
        self.save_hyperparameters() ### if PULSAR LAYER in training_step
    
    def forward(self, x : torch.tensor)-> torch.tensor:
        output = self.painterDensity(x)
        return output
    

    def training_epoch_end(self, outputs):
        """
        outputs is a python list containing the batch_dictionary from each batch
        for the given epoch stacked up against each other. 
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        ##### using tensorboard logger
        self.logger.experiment.add_scalar("Loss", avg_loss,self.current_epoch)
        epoch_dict = {"loss": avg_loss}
        
        my_epoch = trainer.current_epoch + self.init_epoch +1
        self.my_texPts.save_learned_texPts(my_epoch)
        
        # out_root = "/home/user/output/LearnTexturePoints"
        out_root = f"/home/user/output/LearnTexturePoints/{self.person_list[0]}"
        txt_dir = join(out_root,"txtFiles")
        Path(txt_dir).mkdir(parents=True, exist_ok=True)
        txt_fpath = join(txt_dir,"texPts3d.txt")
        txt_line = f"{my_epoch, torch.min(self.points_texture).item(), torch.max(self.points_texture).item(), torch.sum(self.points_texture).item()}"    
        with open(txt_fpath, "a") as f: 
            f.writelines(txt_line+"\n")
            
    
    
    def training_step(self, batch, batch_idx):
        gt_img, gt_mask = batch["image"], batch["mask"] ## "313",(1, 3, 512, 512),(1, 1, 512, 512)
        self.tex_uv, self.points_texture, optim_tex = self.my_texPts.get_texPts_for_training(self.person_list) 
        gamma_uvw = input_mapping_torch(self.points_texture.clone(), self.device)
        # print(f"gamma_uvw: {gamma_uvw.shape, gamma_uvw.is_cuda, gamma_uvw.requires_grad}")
        latent = batch["latent"] ## [bs1, w512, h512, c3] 
        latent = rearrange(latent, "b w h c -> b c w h") ##[bs, w=512, h=512, ch=3]->[1, 3, 512, 512]
        pts_tex = self.tex_uv.clone().unsqueeze(2) ## [bs, n_pts, 2]->[bs, n_pts, 1, 2]
        latent_image = torch.nn.functional.grid_sample(latent, pts_tex, 
                                                     align_corners=True) ### (b=1, c=3, w=n_pts=300000, h=1)
        latent_image = rearrange(latent_image,"b c w h -> b (w h) c")    
        my_lt_dim = latent_image.shape[2]
        # print(f"latent_image.shape:{latent_image.shape}")
        if(my_lt_dim==16):
            latent_image= latent_image[:,:,0:3].clone()    
        # print(f"latent_image.shape:{latent_image.shape}")
        fft_normDirection = batch["fft_normDirection"]
        smpl_poses = batch["smpl_poses"] ## [1, 1, 72]
        smpl = smpl_poses.repeat(1, latent_image.shape[1], 1) ### [1, 300000, 72] 
        ### smpl.shape[2]=72/3 == 24 
        lt_img = torch.repeat_interleave(latent_image, 24, dim=2)### [1, 300000, 72] 
        latent_smpl = lt_img + smpl ### [1, 300000, 72]
        painter_input = torch.cat((gamma_uvw, latent_smpl,fft_normDirection), dim=2).to(torch.float32)  ## [1, 300000, 1096]
        density_input =  torch.cat((gamma_uvw, latent_smpl), dim=2).to(torch.float32)        
        
        out_painter, out_density = self.painterDensity(painter_input, density_input)    
        pulsar_in = torch.cat((out_density, out_painter), dim=2) ## [1, 300000, 33]
        pts_3d, gt_mask  = batch["pts3d"].float(), batch["mask"] ## [bs, n_pts, 3], [bs, ch, h, w]
        vert_radii, rvec = batch["vert_radii"], batch["rvec"] ## [bs, n_pts,1], [bs, 3] 
        Cs, Ks = batch["Cs"], batch["Ks"] ### [bs, 3], [bs, 3, 3]
        pulsar = PulsarLayer(n_pts, height=1024, width=1024,
                            n_channels=33, device=self.device).to(self.device)
        ###[bs1, n_pts300000, ch1] -> [1, 300000]
        out_dense = rearrange(out_density,"b h c -> b (h c)")
        ### opacity: [Bx]N tensor of opacity values in [0., 1.] or None (uses all ones).
        pulsar_out = pulsar(pts_3d, pulsar_in.clone(), vert_radii, rvec, Cs, Ks, opacity=out_dense)
        pulsar_out = rearrange(pulsar_out, "b h w c -> b c h w") ## [1, 32, h, w]
        pulsar_mask = pulsar_out[:, 0, :, :]  ### [1, 1024, 1024]
        pulsar_img = pulsar_out[:, 1:4,:, :]  ### [1, 3, 1024, 1024] (-0.6411585807800293, 0.0) w/ density
        pred_image = pulsar_img * gt_mask  ## [1, 3, 1024, 1024](-0.6411585807800293, 0.0) w/ density
        self.points_texture.retain_grad()
        
        optim = self.optimizers()
        optim_dict= optim.__dict__ ## ['defaults', '_zero_grad_profile_name', 'state', 'param_groups', '_warned_capturable_if_run_uncaptured', '_optimizer', '_strategy', '_optimizer_idx', '_on_before_step', '_on_after_step']        
        # optim_state = optim_dict["state"]  ## defaultdict(<class 'dict'>, {})
        optim_defaults = optim_dict['defaults'] ## ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable']
        # print(f'defaults: {optim_defaults.keys(), optim_defaults}')
        lr = optim_defaults["lr"]
        
        
        epoch = trainer.current_epoch
        out_root = f"/home/user/output/LearnTexturePoints/{self.person_list[0]}_{my_lt_dim}Ch"
        save_dir = join(out_root,"pulsar_images")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        #"Since 1 mesh and All Cam"
        subject_view_dict={"313":1, "315":1,"377":5,"386":8, "387":5,
                           "390":5, "392":5, "393":7, "394":3}
        view_cam = subject_view_dict[self.person_list[0]]
        if (epoch%1==0 and batch_idx==view_cam-1):
            save_fpath = join(save_dir, f"epoch_{epoch}_cam_{view_cam}.png")
            my_img = (pulsar_img[0,:,:,:]+1)/2
            save_image(my_img, save_fpath)
        
          
        ### [0,1] to [-1,1] x_norm = 2 * (x-torch.min(x))/ (torch.max(x)-torch.min(x)) -1
        gt_norm = 2 * (gt_img - torch.min(gt_img)) / (torch.max(gt_img)- torch.min(gt_img))-1  ### (-1.0, 1.0)
        gt_image = gt_norm * gt_mask     ## [1, 3, 1024, 1024],torch.float32, 0.0 , 0.0039 (-1.0, 1.0)
        l1_loss = nn.L1Loss()
        loss_pulsarImg = l1_loss(pred_image, gt_image) ###  (-0.08053641021251678, 1.0), (0.0,1.0)
        
        pulsar_mask = pulsar_mask.unsqueeze(1) ## [1, 1024, 1024]-> [1, 1, 1024, 1024]
        pred_mask = (pulsar_mask-torch.min(pulsar_mask)) / (torch.max(pulsar_mask)- torch.min(pulsar_mask)) ### [1, 1, 1024, 1024] [0,1] 0.0, 1.0
        bce_loss = nn.BCELoss()
        loss_pulsarMask = bce_loss(pred_mask, gt_mask) ### (0.0, 1.0), (0.0, 1.0)
        
        total_loss = loss_pulsarImg + loss_pulsarMask
        
        self.log("total_loss", total_loss, rank_zero_only=True)
        out_dict = {"loss": total_loss, "epoch" : trainer.current_epoch}
        optim.zero_grad()
        self.manual_backward(total_loss)
        optim.step()
        
        return out_dict


    def test_step(self, batch, batch_idx):
        gt_img, gt_mask = batch["image"], batch["mask"] ## "313",(1, 3, 512, 512),(1, 1, 512, 512)
        self.tex_uv, self.points_texture, optim_tex = self.my_texPts.get_texPts_for_training(self.pid) 
        gamma_uvw = input_mapping_torch(self.points_texture.clone(), self.device)
        # print(f"gamma_uvw: {gamma_uvw.shape, gamma_uvw.is_cuda, gamma_uvw.requires_grad}")
        latent = batch["latent"] ## [bs1, w512, h512, c3] 
        latent = rearrange(latent, "b w h c -> b c w h") ##[bs, w=512, h=512, ch=3]->[1, 3, 512, 512]
        pts_tex = self.tex_uv.clone().unsqueeze(2) ## [bs, n_pts, 2]->[bs, n_pts, 1, 2]
        latent_image = torch.nn.functional.grid_sample(latent, pts_tex, 
                                                     align_corners=True) ### (b=1, c=3, w=n_pts=300000, h=1)
        latent_image = rearrange(latent_image,"b c w h -> b (w h) c")    
            
        fft_normDirection = batch["fft_normDirection"]
        smpl_poses = batch["smpl_poses"] ## [1, 1, 72]
        smpl = smpl_poses.repeat(1, latent_image.shape[1], 1) ### [1, 300000, 72] 
        ### smpl.shape[2]=72/3 == 24 
        lt_img = torch.repeat_interleave(latent_image, 24, dim=2)### [1, 300000, 72] 
        latent_smpl = lt_img + smpl ### [1, 300000, 72]
        painter_input = torch.cat((gamma_uvw, latent_smpl,fft_normDirection), dim=2).to(torch.float32)  ## [1, 300000, 1096]
        density_input =  torch.cat((gamma_uvw, latent_smpl), dim=2).to(torch.float32)        
        
        out_painter, out_density = self.painterDensity(painter_input, density_input, batch)    
        pulsar_in = torch.cat((out_density, out_painter), dim=2) ## [1, 300000, 33]
        pts_3d, gt_mask  = batch["pts3d"].float(), batch["mask"] ## [bs, n_pts, 3], [bs, ch, h, w]
        vert_radii, rvec = batch["vert_radii"], batch["rvec"] ## [bs, n_pts,1], [bs, 3] 
        Cs, Ks = batch["Cs"], batch["Ks"] ### [bs, 3], [bs, 3, 3]
        pulsar = PulsarLayer(n_pts, height=1024, width=1024,
                            n_channels=33, device=self.device).to(self.device)
        ###[bs1, n_pts300000, ch1] -> [1, 300000]
        out_dense = rearrange(out_density,"b h c -> b (h c)")
        ### opacity: [Bx]N tensor of opacity values in [0., 1.] or None (uses all ones).
        pulsar_out = pulsar(pts_3d, pulsar_in.clone(), vert_radii, rvec, Cs, Ks, opacity=out_dense)
        pulsar_out = rearrange(pulsar_out, "b h w c -> b c h w") ## [1, 32, h, w]
        pulsar_mask = pulsar_out[:, 0, :, :]  ### [1, 1024, 1024]
        pulsar_img = pulsar_out[:, 1:4,:, :]  ### [1, 3, 1024, 1024] (-0.6411585807800293, 0.0) w/ density
        pred_image = pulsar_img * gt_mask  ## [1, 3, 1024, 1024](-0.6411585807800293, 0.0) w/ density
        print(f"pred_image test: {torch.min(pred_image).item(), torch.max(pred_image)}")
        exit()
       

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {"params": self.painterDensity.parameters(),"lr":0.001, "betas":(0.9, 0.999)},
            {"params":[self.points_texture], "lr":0.001, "betas":(0.9,0.999), "amsgrad":True, "weight_decay":0}
            ])
        return optimizer


class ZJUDataModule(pl.LightningDataModule):
    def __init__(self, root_dir:str, person_ids:list, mesh_nums:list,
                 sampling_points:int=300000, batch_size:int=4,
                 latent_dim:int=3):
        super().__init__()
        self.prepare_data_per_node = True
        self.root_dir = root_dir
        self.person_ids= person_ids
        self.n_pts = sampling_points
        self.bs = batch_size
        self.mesh_nums = mesh_nums
        self.latent_dim = latent_dim
    
    def setup(self, stage:str=None):
        #### stage: "fit", "validate", "test", "predict".
        if stage in (None,"fit"):
            print("@@@@@    CREATE DATASET FOR TRAINING    @@@@@")
            Persons =[]
            Datasets = []
            trans = T.Compose([T.ToTensor(), T.ToPILImage(), 
                               T.Resize(1024),T.ToTensor()])
            # trans = T.Compose([T.ToTensor()])
            for person_id in self.person_ids:
                person = PersonPulsar(person_id, root= self.root_dir,mesh_number=self.mesh_nums, dataset_type=stage)
                person_dataset = PersonDatasetPulsar(person, sampling_points=self.n_pts, 
                                                     transform=trans, latent_dim= self.latent_dim )
                print("person: ", person_id, "dataset length: ", len(person_dataset))
                Datasets.append(person_dataset)
                Persons.append(person)
            self.data_set = ConcatDataset(Datasets)  
        
        elif(stage=="validate"):
            Persons =[]
            Datasets = []
            trans = T.Compose([T.ToTensor(), T.ToPILImage(), 
                               T.Resize(1024),T.ToTensor()])
            # trans = T.Compose([T.ToTensor()])
            for person_id in self.person_ids:
                person = PersonPulsar(person_id, root= self.root_dir,mesh_number=self.mesh_nums, dataset_type=stage)
                person_dataset = PersonDatasetPulsar(person, sampling_points=self.n_pts, 
                                                     transform=trans)
                print("person: ", person_id, "dataset length: ", len(person_dataset))
                Datasets.append(person_dataset)
                Persons.append(person)
            self.data_set = ConcatDataset(Datasets)  
                      
        else: 
            Persons =[]
            Datasets = []
            trans = T.Compose([T.ToTensor(), T.ToPILImage(), T.Resize(1024),T.ToTensor()])
            # trans = T.Compose([T.ToTensor()])
            print("@@@@@    CREATE DATASET FOR  TEST    @@@@@")
            for person_id in self.person_ids:
                person = PersonPulsar(person_id, root= self.root_dir, mesh_number=self.mesh_nums, dataset_type=stage)
                person_dataset = PersonDatasetPulsar(person, sampling_points=self.n_pts,
                                                     transform=trans)
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
    parser.add_argument("--mesh", nargs="+", type=int, help= "Index of the mesh")
    parser.add_argument("--bs", type=int, default=1, action="store", help="Batch Size")
    parser.add_argument("--latent_ch", type=int, default=3, help="Dimension of Latent Codes")
    parser.add_argument("--run_type", type=str, default="training", help="training or testing")
    parser.add_argument("--trainedModel", action="store_true", help="Training From Trained Model")
    parser.add_argument("--newModel", dest="trainedModel", action="store_false")
    parser.set_defaults(trainedModel=False)
    parser.add_argument("--resume_training", action="store_true", help="Resume Training from already trained .ckpt file")
    parser.set_defaults(resume_training=False)
    args = parser.parse_args() ### Namespace(bs=1, run_type='training', trainedModel=False)
    print(f"Arguments From Command Line: \n{args}")
    pid_dict={0:"313",1:"315",2:"377",3:"386",4:"387",5:"390",6:"392",7:"393",8:"394"}
    # person_ids=[0]
    person_idx=args.pid
    person=pid_dict[person_idx]
    print(f"pid from argument: {args.pid}")
    bs = args.bs
    latent_dim= args.latent_ch
    mesh_nums = args.mesh
    num_pose = "SinglePose" if len(mesh_nums)==1 else "MultiPose"
    my_logger = logging.getLogger(__name__)
    my_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"/home/user/output/Logger/logger_texture_{person}.txt")
    fh.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s - %(name)s  - %(levelname)s - %(message)s " )
    fh.setFormatter(formatter)
    my_logger.addHandler(fh)
    
    try:        
        print(f"number of meshes: {len(mesh_nums), num_pose}")
        root_dir = "/home/user/my_zjuMocap"
        out_dir = f"/home/user/LearnTexturePoints_and_LatentCodes/Person_{person}" 
        pl_log_dir = join(out_dir,num_pose, f"TexturePoints_{latent_dim}Ch") 
        print(f"out_dir:{os.path.isdir(out_dir), out_dir}")
        print(f"pl_log_dir: {os.path.isdir(pl_log_dir), pl_log_dir}")
        Path(pl_log_dir).mkdir(parents=True,exist_ok=True)
        training_path= pl_log_dir
        logger = TensorBoardLogger(pl_log_dir, name="pl_model")
        
        n_pts=300000       
        zju_trainLoader = ZJUDataModule(root_dir=root_dir, person_ids=[person_idx],
                                        sampling_points=300000, batch_size=bs,
                                        mesh_nums= mesh_nums,latent_dim=latent_dim)
        # zju_valLoader =  valDataLoader_zju(root_dir=root_dir, person_ids=person_ids,
        #                                 sampling_points=300000, batch_size=bs,
        #                                 mesh_nums= mesh_nums)
        ####==============================================================####        
        #####   PAINTER   NET   #####
        # latent_dim = 3  ## 16
        enc_dim_x = 512 ### After 3D pts/ 3D texture pts encoding with Fourier Feature Mapping
        enc_dim_d = 512 ### View Direction encoded with Fourier Feature Mapping
        latent_smpl_dim = 72 ### latent_image + smpl_poses
        in_dim_painter = enc_dim_x + latent_smpl_dim + enc_dim_d ## 1096
        in_dim_density = enc_dim_x + latent_smpl_dim
        ####==============================================================####        
        
        ################################################
        ######    Painter   And  Density  Network  ####
        ################################################        
        model_path = join(pl_log_dir,"version_NoDir/checkpoints","epoch=49-step=1050.ckpt")  ##E107, pred_img[-1,1], opacity=density 
        
        input_ch, output_ch = 33, 4
        painter_density_net = PainterDensityNet(in_painter= in_dim_painter, in_density= in_dim_density)
        # my_neuralShader = ShallowNeuralShader(input_ch, output_ch)
        init_epoch = -1 ## -1 ## 
        ###================================================================================================###
        load_checkpoint = args.trainedModel   # False #True
        # pids = [pid_dict[i] for i in person_ids] ## pids:['313']
        # pids=[person_ids] ## ['313']
        # exit()
        if load_checkpoint: 
            print("\n ========  Resuming From CheckPoint  ========")            
            pl_net = LitPainterDensity.load_from_checkpoint(checkpoint_path=model_path,
                                                        my_painterDensity=painter_density_net,
                                                        person=person,mesh_num =mesh_nums[0], 
                                                        n_sample=n_pts, save_path=training_path,
                                                        init_epoch=init_epoch)
            
        else:
            print("\n========Running Training From Scratch========")            
            pl_net = LitPainterDensity(painter_density_net, person, mesh_nums[0],
                                       n_pts, training_path, init_epoch)
        ## ============================================================================#####
        ### Trainer Class API https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#methods
        ### strategy="ddp_find_unused_parameters_false", "horovod"
        if(args.run_type=="training"): 
            if args.resume_training:
                print("Resuming From Checkpoint (PL argument)")
                trainer = pl.Trainer(devices=1, accelerator="gpu",
                                     limit_train_batches=1.0, max_epochs=100, 
                                     default_root_dir=training_path,
                                     logger=logger, resume_from_checkpoint=model_path,
                                     callbacks=[MyPrintingCallback()])
            else:
                print("Normal Training")
                saveCkptPerEpoch = CheckpointEveryNEpochs(10)
                trainer = pl.Trainer(devices=1, accelerator="gpu",
                                     limit_train_batches=1.0, max_epochs=100, 
                                     default_root_dir=training_path,
                                     logger=logger,
                                     callbacks=[MyPrintingCallback()])
                                                # saveCkptPerEpoch])
            trainer.fit(pl_net, zju_trainLoader)#, zju_valLoader)
            
        elif(args.run_type=="testing"):
            #########    TEST    (Novel-View Synthesis)   #######        
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


# from ast import main
import argparse
import os
import warnings
from audioop import avg
from os.path import join
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from einops import rearrange
from numpy.random import default_rng
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

# import torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.utils import save_image

from dataset_pulsar import PersonDatasetPulsar, PersonPulsar
from impCarv_points import PulsarLayer
from latent_codes import LatentCodes

warnings.filterwarnings("ignore")
import logging
import smtplib
import ssl
import time
from email.message import EmailMessage


def email_for_error(body=" "):
    username = "pallab38@gmail.com"
    password = "bcuxcgbdolibyqvq"
    receiver = "s6padass@uni-bonn.de"

    em = EmailMessage()
    em["From"] = username
    em["To"] = receiver
    em["Subject"] = "!!! ERORR On the CVG-SRV07 server !!!"
    em.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(username, password)
        smtp.sendmail(username, receiver, em.as_string())

    print("Email has sent")


def send_email(epoch_end=True, epoch_num=0, error=False):
    username = "pallab38@gmail.com"
    password = "bcuxcgbdolibyqvq"
    receiver = "s6padass@uni-bonn.de"

    em = EmailMessage()
    em["From"] = username
    em["To"] = receiver
    em["Subject"] = "Status of Running Processes On the CVG-SRV07 server"
    if epoch_end:
        body = f"The epoch {epoch_num} is done. "
    else:
        body = "The training is finished"

    em.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(username, password)
        smtp.sendmail(username, receiver, em.as_string())


def init_processes(rank, size, fn, my_args, my_queue, backend="gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29560"
    distributed.init_process_group(backend, rank=rank, world_size=size)

    out = fn(
        my_args["pts"],
        my_args["lt_img"],
        my_args["rad"],
        my_args["r"],
        my_args["cs"],
        my_args["ks"],
    )
    my_queue.put(out.detach())


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("->>>>>>>  Training is starting   <<<<<<<-")

    def on_train_end(self, trainer, pl_module):
        print("->>>>>>>  Training is ending  <<<<<<<-")
        # send_email(epoch_end=False)


class LitUNet(pl.LightningModule):
    def __init__(self, train_path, latent_dim, person, bs=1, latent_texture_size=512):
        super().__init__()
        # self.unet = unet ## for 3 channels NO UNet
        self.train_path = train_path
        self.latent_dim = latent_dim
        self.person = person
        # self.person_idx = person_idx
        self.batch_size = bs
        self.latent_texture_size = latent_texture_size
        self.save_hyperparameters()
        # person_all = [self.person_id]
        print(f"LitUNet->person_all:{[self.person]}")
        self.latent_codes = LatentCodes(
            self.latent_dim,
            self.train_path,
            [self.person],
            -1,
            self.device,
            latent_texture_size=self.latent_texture_size,
        )
        self.latent_dict = self.latent_codes.data[self.person]
        self.latent = self.latent_dict["latent"]
        # self.latent_image = self.latent_dict["latent_image"]

    def on_epoch_end(self):
        # my_epoch = trainer.current_epoch
        # self.latent_codes.save_all_codes(my_epoch)
        # send_email(epoch_end =True,epoch_num=my_epoch)
        # print(f"on_epoch_end-> outputs.keys(): {outputs.keys()}")
        pass

    def training_epoch_end(self, outputs):
        """
        outputs is a python list containing the batch_dictionary from each batch
        for the given epoch stacked up against each other.
        """

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # print(f"avg_loss: {avg_loss}")
        self.logger.experiment.add_scalar("Loss", avg_loss, self.current_epoch)
        epoch_dict = {"loss": avg_loss}

        out_dict = outputs[1]
        # print(f"out_dict: {type(out_dict)}, keys: {out_dict.keys()}")
        # print(f"self.latent: {torch.sum(self.latent)}")
        # print(f"self.latent_image: {torch.sum(self.latent_image)}")
        # self.latent = out_dict["lt"]
        latent_image = out_dict["lt_img"]
        my_epoch = trainer.current_epoch
        self.latent_codes.save_all_codes(latent_image, my_epoch)

    def configure_optimizers(self):
        optimizer = optim.Adam([self.latent], lr=0.001, amsgrad=True, weight_decay=0)
        return optimizer

    def training_step(self, batch, batch_idx):
        ## self.trainer.global_step, self.trainer.max_epochs

        pts3d, pts_uv, normal_noise = (
            batch["pts3d"].float(),
            batch["pts_uv"].float(),
            batch["normal_noise"],
        )  ## [1, 300000, 3]  [1, 300000, 2]
        pid, gt_img, gt_mask = (
            batch["pid"],
            batch["image"],
            batch["mask"],
        )  ## "313",(1, 3, 512, 512),(1, 1, 512, 512)
        self.width, self.height = (
            gt_img.shape[2],
            gt_img.shape[2],
        )  ## width: 1024, height:1024
        rvec, Cs, Ks = (
            batch["rvec"],
            batch["Cs"],
            batch["Ks"],
        )  #### rvec: torch.Size([1, 3]), Cs: torch.Size([1, 3]), Ks: torch.Size([1, 3, 3])
        vert_radii, normals = (
            batch["vert_radii"],
            batch["normals"].detach().cpu().numpy(),
        )  ## ###(n_pts x 1)  (batch_size x n_pts x 3)

        n_samples = pts3d.shape[1]
        rng = default_rng(12345)
        noise_fixed = rng.uniform(-0.1, 0.1, (n_samples, 1))
        normal_vec = normals * noise_fixed
        pts3d = pts3d + torch.from_numpy(normal_vec).to(self.device)
        pts_uv = pts_uv.unsqueeze(2)  ## (b=1, n_pts=300000, fake_dim=1, 2)
        self.latent, optims = self.latent_codes.get_codes_for_training(
            list(pid)
        )  ###[bs=1,w=512,h=512,ch=3]
        latent = rearrange(
            self.latent, "b w h c-> b c w h"
        )  ### (b=1, c=3, w=1024, h=1024)
        latent_img = torch.nn.functional.grid_sample(
            latent.to(self.device), pts_uv, align_corners=True
        )  ### (b=1, c=3, w=n_pts=300000, h=1)
        latent_img = rearrange(
            latent_img, "b c w h -> b (w h) c"
        )  ### (b=1, w=n_pts=300000, c=3)->[1, 3, 300000]

        n_pts = latent_img.shape[1]
        n_channels = latent_img.shape[2]
        pulsar = PulsarLayer(
            n_pts, self.height, self.width, n_channels, device=self.device
        ).to(self.device)
        pts3d = pts3d.float().to(self.device)
        img_hat = pulsar(
            pts3d, latent_img, vert_radii, rvec, Cs, Ks
        )  ##[1, 1024, 1024, 3]
        img_hat = rearrange(
            img_hat, "b w h c -> b c w h"
        )  ##[1, 1024, 1024, 3] -> ]1, 3, 1024, 1024
        if self.latent_dim == 16:
            img_hat = img_hat[:, 0:3, :, :]  ## [1,3,1024,1024]
            # print(f"latent 16 img_hat:{img_hat.shape}")
        save_dir = f"/home/user/output/LatentCodes/{self.person}_{self.latent_dim}Ch"
        pulsar_dir = join(save_dir, "pulsar_images")
        Path(pulsar_dir).mkdir(parents=True, exist_ok=True)
        txt_dir = join(save_dir, "txtFiles")
        Path(txt_dir).mkdir(parents=True, exist_ok=True)
        txt_fpath = join(txt_dir, "latent_grads.txt")

        # "Since 1 mesh and All Cam"
        subject_view_dict = {
            "313": 1,
            "315": 1,
            "377": 5,
            "386": 8,
            "387": 5,
            "390": 5,
            "392": 5,
            "393": 7,
            "394": 3,
        }
        view_cam = subject_view_dict[self.person]
        if batch_idx == view_cam - 1:
            lt_grad_line = f"latent sum: {torch.sum(self.latent)}"
            with open(txt_fpath, "a") as f:
                f.writelines(lt_grad_line + "\n")
            img_pulsar = img_hat[0, :, :, :]
            img_path = join(
                pulsar_dir,
                "epoch_%03d_cam_%02d.png" % (trainer.current_epoch, view_cam),
            )
            save_image(img_pulsar, img_path)
            # exit()
        l1_loss = nn.L1Loss()
        loss_img = l1_loss(img_hat, gt_img)
        self.log("loss", loss_img, on_epoch=True, prog_bar=True, on_step=True)
        out_dict = {"loss": loss_img, "lt_img": latent_img, "lt": latent}
        return out_dict


class ZJUDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        person_ids: list,
        sampling_points: int = 300000,
        batch_size: int = 4,
    ):
        super().__init__()
        self.prepare_data_per_node = True
        self.root_dir = root_dir
        self.person_ids = person_ids
        self.n_pts = sampling_points
        self.bs = batch_size

    def setup(self, stage: str = None):
        if stage in (None, "fit"):
            Persons = []
            Datasets = []
            trans = T.Compose(
                [T.ToTensor(), T.ToPILImage(), T.Resize(1024), T.ToTensor()]
            )
            print(f"zju person_ids:{self.person_ids}")
            for person_id in self.person_ids:
                print(f"zjudata:{person_id},{type(person_id)}")
                person = PersonPulsar(person_id, mesh_number=[0], root=self.root_dir)
                person_dataset = PersonDatasetPulsar(
                    person, sampling_points=self.n_pts, transform=trans
                )
                print("person: ", person_id, "dataset length: ", len(person_dataset))
                Datasets.append(person_dataset)
                Persons.append(person)
            self.data_set = ConcatDataset(Datasets)

    def train_dataloader(self):
        zju_train = DataLoader(
            self.data_set,
            batch_size=self.bs,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
            drop_last=True,
        )  #
        total_frames = len(self.data_set)
        print(
            "Total : {0}, batch: {1}, batch_size:{2}".format(
                total_frames, len(zju_train), self.bs
            )
        )
        return zju_train


if __name__ == "__main__":
    pl.utilities.seed.seed_everything(1310)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, default=0, help="Pid:0 for 313 or 2 for 377")
    parser.add_argument("--bs", type=int, default=1, action="store", help="Batch Size")
    parser.add_argument(
        "--latent_ch", type=int, default=3, help="Dimension of Latent Codes"
    )
    args = parser.parse_args()  ### Namespace(bs=1, pid=2)
    print(f"Arguments From Command Line: \n{args}")
    pid_dict = {
        0: "313",
        1: "315",
        2: "377",
        3: "386",
        4: "387",
        5: "390",
        6: "392",
        7: "393",
        8: "394",
    }
    # person_ids=[0]
    # bs = 1
    person_idx = args.pid  ### 313 or 377
    person = pid_dict[person_idx]
    bs = args.bs
    latent_dim = args.latent_ch
    output_dirpath = "/home/user/output/Logger"
    Path(output_dirpath).mkdir(parents=True, exist_ok=True)
    my_logger = logging.getLogger(__name__)
    my_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(join(output_dirpath, f"logger_Latent_{person}.txt"))
    fh.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s  - %(levelname)s - %(message)s "
    )
    fh.setFormatter(formatter)
    my_logger.addHandler(fh)

    try:
        root_dir = "/home/user/my_zjuMocap"
        out_dir = f"/home/user/LearnTexturePoints_and_LatentCodes/Person_{person}"
        pl_log_dir = join(out_dir, "SinglePose", f"LatentCodes_{latent_dim}Ch")
        Path(pl_log_dir).mkdir(parents=True, exist_ok=True)
        training_path = pl_log_dir
        logger = TensorBoardLogger(pl_log_dir, name="pl_model")

        zju_trainLoader = ZJUDataModule(
            root_dir=root_dir,
            person_ids=[person_idx],
            sampling_points=300000,
            batch_size=bs,
        )
        # latent_dim = 3  ## 16
        latent_texture_size = 512
        print(f"person_id: {person}")
        my_latentModel = LitUNet(
            training_path,
            latent_dim=latent_dim,
            person=person,
            bs=bs,
            latent_texture_size=latent_texture_size,
        )

        ## ============================================================================#####
        ### Trainer Class API https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#methods
        # checkpoint_callback = ModelCheckpoint(monitor="loss")
        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu",
            limit_train_batches=1.0,
            max_epochs=100,
            default_root_dir=training_path,
            logger=logger,
            callbacks=[MyPrintingCallback()],
        )
        trainer.fit(
            model=my_latentModel, train_dataloaders=zju_trainLoader
        )  ### for 3 channels NO UNet
    except Exception as e:
        my_logger.exception(e)
        time.sleep(3)
        print(e)
        print(type(e))
        # email_for_error(body=str(e))

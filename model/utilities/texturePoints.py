import os
from os.path import join, isfile, isdir
from pathlib import Path

import numpy as np
from numpy.random import default_rng
import torch
import trimesh

from typing import List
from utilities.nara_processing_surface_new import TextureProjector3d
from utilities.FourierFeature import input_mapping

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


class TexturePoints3D:
    def __init__(
        self,
        person_ids: List[str],
        mesh_num: int,
        n_sample: int,
        current_epoch: int,
        save_path: str,
        device: torch.device,
        data_root: str = "/home/user/my_zjuMocap",
    ) -> None:
        """Learns The 3D Texture Points (u x h x w).
        **Texture Points** are w/o batch.

        Args:
            person_ids (List[str]): List of person ids, "313"
            mesh_num (int): Mesh number, 0
            n_sample (int): Number of Sample Points, 300K
            current_epoch (int): Current Epoch, -1 for initialization
            save_path (str): Where to save Learned 3D textures
            device (torch.device): Cuda or CPU
            data_root (str, optional): Directory for Mesh, UV Table. Defaults to "/home/user/my_zjuMocap".
        Returns:

        """
        super().__init__()
        self.person_ids = person_ids
        self.mesh_num = mesh_num
        self.n_sample = n_sample
        self.current_epoch = current_epoch
        self.device = device
        self.save_dir = join(save_path, "learned_texPts")
        self.mesh_dir = join(data_root, "50_meshes")
        self.uv_fpath = join(data_root, "uv_table.npy")
        self.data = dict()
        if not isdir(self.save_dir):
            assert (
                current_epoch == -1
            ), f"Requires Learned Texture Points for {self.current_epoch} at {self.save_dir}"
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        if self.current_epoch == -1:
            ###############################################################
            ########    Create Initial Random 3d Texture Points    ########
            ###############################################################
            torch.manual_seed(1310)
            self.rng = default_rng(12345)
            base_tex3d = torch.FloatTensor(
                self.n_sample, 3
            ).uniform_()  ## torch.float32, [300000, 3])
            # print(f"base_tex:{type(base_tex3d), base_tex3d.dtype, base_tex3d.shape}")
            # print(f"base_tex: {torch.min(base_tex3d).item(),torch.max(base_tex3d).item()}")
            person_id = self.person_ids[0]
            mesh_fpath = join(self.mesh_dir, person_id, "%06d.obj" % self.mesh_num)
            my_mesh = trimesh.load(mesh_fpath)
            V = my_mesh.vertices
            F = my_mesh.faces
            T = np.load(self.uv_fpath)

            proj = TextureProjector3d(T, F=F)
            pts3d, pts_uv, normal_noise, noramls = proj.random_sample(
                n_sample, V
            )  ### (300000, 2) 0.01966 0.9945
            ### Convert Value Range: [0,1]->[-1,1]
            pts_uv = (
                2.0 * (pts_uv - np.min(pts_uv)) / np.ptp(pts_uv) - 1
            )  ## (300000, 2), -1.0, 1.0)
            noise_fixed = self.rng.uniform(
                -0.1, 0.1, (self.n_sample, 1)
            )  ## (300000, 1)
            uvw = np.concatenate((pts_uv, noise_fixed), axis=1)  ## (300000, 3)
            base_tex3d = torch.from_numpy(
                uvw.copy()
            ).float()  ## torch.float32, [300000, 3])
            # print(f"base_tex:{type(base_tex3d), base_tex3d.dtype, base_tex3d.shape}")

            # print(f"base_tex: {torch.min(base_tex3d).item(),torch.max(base_tex3d).item()}")
            base_fpath = join(self.save_dir, "base.pth")
            torch.save(base_tex3d, base_fpath)

            for pid in self.person_ids:
                ##### Save the Texture Points ####
                tex_pts = torch.load(base_fpath).to(self.device)
                tex_pts.requires_grad = True
                optim = torch.optim.Adam(
                    [tex_pts], lr=0.05, amsgrad=True, weight_decay=0
                )
                self.data[pid] = {"tex_pts": tex_pts, "optim": optim}

        else:
            ###############################################################
            ########       Load Learned 3d Texture Points          ########
            ###############################################################
            path = join(self.save_dir, "ep%06d" % current_epoch)
            for pid in self.person_ids:
                tex_fpath = join(path, f"texture_{pid}.pth")
                optim_fpath = join(path, f"texture_{pid}_optim.pth")
                assert isfile(tex_fpath), tex_fpath
                assert isfile(optim_fpath), optim_fpath
                tex_pts = torch.load(tex_fpath).to(self.device)
                tex_pts.requires_grad = True
                optim = torch.optim.Adam(
                    [tex_pts], lr=0.005, amsgrad=True, weight_decay=0
                )
                optim_checkpoint = torch.load(optim_fpath)
                optim.load_state_dict(optim_checkpoint["optim"])
                self.data[pid] = {"tex_pts": tex_pts, "optim": optim}

    def save_learned_texPts(self, epoch=-1):
        if epoch > -1:
            epoch_dir = join(self.save_dir, "ep%06d" % epoch)
            assert not isdir(epoch_dir), epoch_dir
            Path(epoch_dir).mkdir(parents=True, exist_ok=True)

        for pid in self.data.keys():
            data = self.data[pid]
            tex_pts = data["tex_pts"]
            optim = data["optim"]
            tex_fpath = join(self.save_dir, f"texture_{pid}.pth")
            optim_fpath = join(self.save_dir, f"texture_{pid}_optim.pth")
            torch.save({"optim": optim.state_dict()}, optim_fpath)
            torch.save(tex_pts, tex_fpath)
            # print("tex_pts is saved in :", tex_fpath)

            if epoch > -1:
                f_path = join(epoch_dir, f"texture_{pid}.pth")
                optim_path = join(epoch_dir, f"texture_{pid}_optim.pth")
                torch.save({"optim": optim.state_dict()}, optim_path)
                torch.save(tex_pts, f_path)

    def get_texPts_for_evaluation(self, person_ids: List[int]):
        texPts_3d = []
        for pid in person_ids:
            data = self.data[pid]
            tex_pts = data["tex_pts"]
            texPts_3d.append(tex_pts.unsqueeze(0))
        texPts_3d = torch.cat(texPts_3d, dim=0)
        return texPts_3d

    def get_texPts_for_training(self, person_ids: List[int]):
        """
        Returns 3D Texture Points and it's respective Optimizer
            Person can be repeated- then only Optimizer will return

        Args:
            person_ids (List[int]): _description_
        """
        texPts_3d = []
        Optim = []
        already_used_pids = set()
        for pid in person_ids:
            data = self.data[pid]
            tex_pts = data["tex_pts"]
            # print(f"{type(tex_pts), tex_pts.requires_grad}")
            optim = data["optim"]
            texPts_3d.append(tex_pts.unsqueeze(0))

            if pid not in already_used_pids:
                Optim.append(optim)
                already_used_pids.add(pid)
        texPts_3d = torch.cat(texPts_3d, dim=0)
        return texPts_3d, Optim


class TexturePoints:
    def __init__(
        self,
        person_ids: List[str],
        mesh_num: int,
        n_sample: int,
        current_epoch: int,
        save_path: str,
        device: torch.device,
        data_root: str = "/home/user/my_zjuMocap",
    ) -> None:
        """Learns The 3D Texture Points (u x h x w).
        **Texture Points** are w/o batch.

        Args:
            person_ids (List[str]): List of person ids, "313"
            mesh_num (int): Mesh number, 0
            n_sample (int): Number of Sample Points, 300K
            current_epoch (int): Current Epoch, -1 for initialization
            save_path (str): Where to save Learned 3D textures
            device (torch.device): Cuda or CPU
            data_root (str, optional): Directory for Mesh, UV Table. Defaults to "/home/user/my_zjuMocap".
        Returns:

        """
        super().__init__()
        self.person_ids = person_ids
        self.mesh_num = mesh_num
        self.n_sample = n_sample
        self.current_epoch = current_epoch
        self.device = device
        self.save_dir = join(save_path, "learned_texPts")
        self.mesh_dir = join(data_root, "50_meshes")
        self.uv_fpath = join(data_root, "uv_table.npy")
        self.data = dict()
        if not isdir(self.save_dir):
            assert (
                current_epoch == -1
            ), f"Requires Learned Texture Points for {self.current_epoch} at {self.save_dir}"
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        if self.current_epoch == -1:
            ###############################################################
            ########    Create Initial Random 3d Texture Points    ########
            ###############################################################
            torch.manual_seed(1310)
            self.rng = default_rng(12345)
            base_tex3d = (
                torch.FloatTensor(self.n_sample, 3).uniform_().to(self.device)
            )  ## torch.float32, [300000, 3])
            base_tex2d = torch.FloatTensor(self.n_sample, 2).uniform_().to(self.device)
            # print(f"base_tex:{type(base_tex3d), base_tex3d.dtype, base_tex3d.shape}")
            # print(f"base_tex: {torch.min(base_tex3d).item(),torch.max(base_tex3d).item()}")
            person_id = self.person_ids[0]
            mesh_fpath = join(self.mesh_dir, person_id, "%06d.obj" % self.mesh_num)
            my_mesh = trimesh.load(mesh_fpath)
            V = my_mesh.vertices
            F = my_mesh.faces
            T = np.load(self.uv_fpath)

            proj = TextureProjector3d(T, F=F)
            pts3d, pts_uv, normal_noise, noramls = proj.random_sample(
                n_sample, V
            )  ### (300000, 2) 0.01966 0.9945
            ### Convert Value Range: [0,1]->[-1,1]
            pts_uv = (
                2.0 * (pts_uv - np.min(pts_uv)) / np.ptp(pts_uv) - 1
            )  ## (300000, 2), -1.0, 1.0)
            noise_fixed = self.rng.uniform(
                -0.1, 0.1, (self.n_sample, 1)
            )  ## (300000, 1)
            uvw = np.concatenate((pts_uv, noise_fixed), axis=1)  ## (300000, 3)
            base_tex3d = torch.from_numpy(
                uvw.copy()
            ).float()  ## torch.float32, [300000, 3])
            # print(f"base_tex: {torch.min(base_tex3d).item(),torch.max(base_tex3d).item()}")
            base_tex2d = torch.from_numpy(pts_uv.copy()).float()

            base3d_fpath = join(self.save_dir, "base_3d.pth")
            torch.save(base_tex3d, base3d_fpath)
            base2d_fpath = join(self.save_dir, "base_2d.pth")
            torch.save(base_tex2d, base2d_fpath)
            for pid in self.person_ids:
                ##### Load the Texture Points ####
                tex_2d = torch.load(base2d_fpath).to(self.device)
                tex_2d.requires_grad = True
                tex_pts = torch.load(base3d_fpath).to(self.device)
                tex_pts.requires_grad = True
                optim = torch.optim.Adam(
                    [tex_pts, tex_2d], lr=0.05, amsgrad=True, weight_decay=0
                )
                self.data[pid] = {"tex_2d": tex_2d, "tex_pts": tex_pts, "optim": optim}

        else:
            ###############################################################
            ########       Load Learned 3d Texture Points          ########
            ###############################################################

            path = join(self.save_dir, "ep%06d" % current_epoch)
            for pid in self.person_ids:
                tex2d_fpath = join(path, f"texture2d_{pid}.pth")
                tex_fpath = join(path, f"texture3d_{pid}.pth")
                optim_fpath = join(path, f"texture_{pid}_optim.pth")
                assert isfile(tex2d_fpath), tex2d_fpath
                assert isfile(tex_fpath), tex_fpath
                assert isfile(optim_fpath), optim_fpath
                tex_2d = torch.load(tex2d_fpath).to(self.device)
                tex_2d.requires_grad = True
                tex_pts = torch.load(tex_fpath).to(self.device)
                tex_pts.requires_grad = True
                optim = torch.optim.Adam(
                    [tex_pts], lr=0.005, amsgrad=True, weight_decay=0
                )
                optim_checkpoint = torch.load(optim_fpath)
                optim.load_state_dict(optim_checkpoint["optim"])
                self.data[pid] = {"tex_2d": tex_2d, "tex_pts": tex_pts, "optim": optim}

    def save_learned_texPts(self, epoch=-1):
        if epoch > -1:
            epoch_dir = join(self.save_dir, "ep%06d" % epoch)
            assert not isdir(epoch_dir), epoch_dir
            Path(epoch_dir).mkdir(parents=True, exist_ok=True)

        for pid in self.data.keys():
            data = self.data[pid]
            tex_2d = data["tex_2d"]
            tex_pts = data["tex_pts"]
            optim = data["optim"]
            tex2d_fpath = join(self.save_dir, f"texture2d_{pid}.pth")
            tex_fpath = join(self.save_dir, f"texture3d_{pid}.pth")
            optim_fpath = join(self.save_dir, f"texture_{pid}_optim.pth")
            torch.save({"optim": optim.state_dict()}, optim_fpath)
            torch.save(tex_2d, tex2d_fpath)
            torch.save(tex_pts, tex_fpath)
            # print("tex_pts is saved in :", tex_fpath)

            if epoch > -1:
                tex2d_path = join(epoch_dir, f"texture2d_{pid}.pth")
                f_path = join(epoch_dir, f"texture3d_{pid}.pth")
                optim_path = join(epoch_dir, f"texture_{pid}_optim.pth")
                torch.save({"optim": optim.state_dict()}, optim_path)
                torch.save(tex_pts, f_path)
                torch.save(tex_2d, tex2d_path)

    def get_texPts_for_evaluation(self, person_ids: List[int]):
        texPts_3d = []
        texPts_2d = []
        for pid in person_ids:
            data = self.data[pid]
            tex_2d = data["tex_2d"]
            tex_pts = data["tex_pts"]
            texPts_2d.append(tex_2d.unsqueeze(0))
            texPts_3d.append(tex_pts.unsqueeze(0))
        texPts_2d = torch.cat(tex_2d, dim=0)
        texPts_3d = torch.cat(texPts_3d, dim=0)
        return texPts_2d, texPts_3d

    def get_texPts_for_training(self, person_ids: List[int]):
        """
        Returns 3D Texture Points and it's respective Optimizer
            Person can be repeated- then only Optimizer will return

        Args:
            person_ids (List[int]): _description_
        """
        texPts_3d = []
        texPts_2d = []
        Optim = []
        already_used_pids = set()
        for pid in person_ids:
            data = self.data[pid]
            tex_2d = data["tex_2d"]
            tex_pts = data["tex_pts"]
            # print(f"{type(tex_pts), tex_pts.requires_grad}")
            optim = data["optim"]
            texPts_2d.append(tex_2d.unsqueeze(0))
            texPts_3d.append(tex_pts.unsqueeze(0))

            if pid not in already_used_pids:
                Optim.append(optim)
                already_used_pids.add(pid)
        texPts_2d = torch.cat(texPts_2d, dim=0).to(self.device)
        texPts_3d = torch.cat(texPts_3d, dim=0).to(self.device)
        # print(f"Texture points CUDA:{texPts_2d.is_cuda, texPts_3d.is_cuda}")
        return texPts_2d, texPts_3d, Optim


if __name__ == "__main__":
    np.random.seed(1310)
    root = "/home/user/DATA"
    person_ids = [0]
    mesh_num = 0
    n_sample = 300000
    currentEpoch = -1
    save_path = join(root, "toDelete")
    device = "cuda:0" if torch.cuda.is_available else "cpu"
    rng = default_rng(12345)
    noise_fixed = rng.uniform(-0.1, 0.1, (1, n_sample, 1))
    print(f"noise_fix: {noise_fixed.shape}")
    texture_points = TexturePoints3D(
        person_ids, mesh_num, n_sample, currentEpoch, save_path, device
    )

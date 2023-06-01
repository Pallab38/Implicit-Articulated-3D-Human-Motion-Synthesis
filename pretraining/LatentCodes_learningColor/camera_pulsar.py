"""
This script(camera class) intended to be used with
PULSAR
"""

import json
import os
from os.path import join

import cv2
import imageio
import numpy as np
import torch

### For the main Function
import trimesh

from impCarv_points import PulsarLayer
from nara_processing_surface_new import TextureProjector3d


class MyCamera:
    @staticmethod
    def load_cam_data(cam_path, cam_list=[]):
        if len(os.listdir(cam_path)) == 1:
            cam_fpath = join(cam_path, "annots.npy")
            if cam_fpath.split(".")[-1] == "json":
                with open(cam_fpath, "r") as f:
                    my_cam_dict = json.load(f)
                    my_cam = my_cam_dict["cams"][
                        "20190823"
                    ]  ### dict_keys(['K', 'R', 'T', 'D'])
            else:
                my_cam_dict = np.load(cam_fpath, allow_pickle=True).item()
                my_cam = my_cam_dict["cams"]  ### dict_keys(['K', 'R', 'T', 'D'])
                # print(my_cam.keys())
            k_vals = my_cam["K"]  ## 21
            r_vals = my_cam["R"]  ## 21
            t_vals = my_cam["T"]  ## 21
            d_vals = my_cam["D"]  ## 21
            if len(cam_list) == 0:
                cam_list = [i for i in range(len(k_vals))]
            # print(cam_list)
            # new_cam_list=[]
            cameras = {}
            P = {}  ## projection matrix
            for cam_num in cam_list:
                camera_id = cam_num
                # new_cam_list.append(camera_id)
                cam = {}  ## Dictionary for individual camera
                cam["K"] = np.array(k_vals[cam_num])  ### 3x3 np.array
                cam["inv_k"] = np.linalg.inv(cam["K"])  ### 3x3 np.array
                cam["dist"] = np.array(d_vals[cam_num])  ### 5x1 np.array

                r_mat = np.array(r_vals[cam_num])
                rvec = cv2.Rodrigues(r_mat)[0]  ### 3x1 np.array
                t_vec = np.array(t_vals[cam_num])  ### 3x1 np.array
                # r_mat = cv2.Rodrigues(r_vec)[0] ### 3x3 np.array
                # r_mat = r_vec
                rt_mat = np.hstack((r_mat, t_vec))  ### 3x4 np.array

                cam["rvec"] = rvec
                cam["R"] = r_mat
                cam["R"] = cam["R"].T.copy()
                cam["T"] = t_vec / 1000.0
                cam["RT"] = rt_mat
                P[cam_num] = cam["K"] @ cam["RT"]  ###  3x4 np.array
                cam["P"] = P[cam_num]

                if camera_id == 19 or camera_id == 20:
                    cameras[camera_id + 3] = cam
                else:
                    cameras[camera_id + 1] = cam
                # cameras[cam_num] = cam
            # cameras['cam_names']= cam_list ## don't need for now
            # cameras['cam_names']= new_cam_list

        elif len(os.listdir(cam_path)) == 2:
            intri_fpath = join(cam_path, "intri.yml")
            extri_fpath = join(cam_path, "extri.yml")
            intri_param = cv2.FileStorage(
                intri_fpath, flags=0
            )  # 0: cv2.FILE_STORAGE_READ; 1: cv2.FILE_STORAGE_WRITE
            extri_param = cv2.FileStorage(extri_fpath, flags=0)
            cam_names = intri_param.getNode("names")
            cam_list = [cam_names.at(i).string() for i in range(cam_names.size())]
            # print(cam_list)
            cam_ids = [k.split("_")[-1] for k in cam_list]
            cam_names = [l.split("B")[1] for l in cam_ids]
            cameras = {}
            P = {}
            for c_name in cam_names:
                j = "Camera_B" + c_name
                cam = {}
                cam["K"] = intri_param.getNode(f"K_{j}").mat()  # 3x3 np.array
                cam["inv_k"] = np.linalg.inv(cam["K"])  # 3x3 np.array
                cam["dist"] = intri_param.getNode(f"dist_{j}").mat()  # 5x1 np.array

                r_vec = extri_param.getNode(f"R_{j}").mat()  # 3x1 np.array
                t_vec = extri_param.getNode(f"T_{j}").mat()  # 3x1 np.array
                r_mat = cv2.Rodrigues(r_vec)[0]  # 3x3 np.array
                rt_mat = np.hstack((r_mat, t_vec))  # 3x4 np.array

                cam["rvec"] = r_vec
                # cam["R"]= r_mat## old
                cam["R"] = r_mat.T.copy()
                cam["T"] = t_vec

                cam["RT"] = rt_mat
                P[j] = cam["K"] @ cam["RT"]  #  3x4 np.array
                cam["P"] = P[j]

                cameras[int(c_name)] = cam

            # cameras["cam_names"] = cam_list
        else:
            print("please check the camera directory: ", cam_path)
        return cameras

    # def __init__(self,K,R,T,P,cid):
    def __init__(self, cameras_dict, cid):
        self.cameras_dict = cameras_dict
        self.cid = cid
        self.my_camera = self.cameras_dict[self.cid]
        self.K = self.my_camera["K"]
        self.R = self.my_camera["R"]
        self.T = self.my_camera["T"]
        self.P = self.my_camera["P"]
        self.rvec = self.my_camera["rvec"]
        self.dist = self.my_camera["dist"]

    def __str__(self):
        txt = f"~~~~~\ncid={self.cid}\n"
        txt += "rvec=" + str(np.squeeze(self.rvec)) + "\n"
        txt += "pos=" + str(np.squeeze(self.C)) + "\n"
        txt += "K=" + str(self.K) + "\n"
        txt += "~~~~~\n"

        return txt

    def cam_for_pulsar(self):
        """
        Returns:
        rvecs: {n_batch x 3}
        Cs: {n_batch x 3}
        Ks: {n_batch x 3 x 3}
        """
        K = torch.from_numpy(self.K).float()
        rvec = torch.squeeze(torch.from_numpy(cv2.Rodrigues(self.R)[0])).float()
        ## For person 313, 315 and 377 ###
        Cs = torch.squeeze(torch.from_numpy(-self.R @ self.T).float())

        ### For rest of the person
        ## C = -R^-1 @ t
        # Cs = torch.squeeze(torch.from_numpy(-np.linalg.inv(self.R) @ self.T).float())
        return K, rvec, Cs

    def param_for_nara(self):
        return self.rvec, self.T, self.K, self.dist


if __name__ == "__main__":
    data_root = "/home/user/my_zjuMocap"
    cam_root = join(data_root, "cameras")
    # person="313" ### For 313 and 315 num_view = 20 and 21 are not available
    person = "386"
    num_view = 1
    frame = 120

    cam_dir = join(cam_root, person)
    # print("len(os.listdir(cam_dir)):",len(os.listdir(cam_dir)))
    all_cameras = MyCamera.load_cam_data(cam_dir)
    # print(all_cameras.keys())
    my_cam = MyCamera(all_cameras, num_view)
    K, rvec, Cs = my_cam.cam_for_pulsar()
    # print(K.shape,rvec.shape,cam_pos.shape)
    print(my_cam.rvec)

    exit()

    ### add dummy batch dimensions
    K = K[None, ...]
    rvec = rvec[None, ...]
    Cs = Cs[None, ...]
    print(K.shape, rvec.shape, Cs.shape)
    ### torch.Size([1, 3, 3]) torch.Size([1, 3]) torch.Size([1, 3])

    ######          GET         MESH         #####
    mesh_root = join(data_root, "meshes")

    mesh_fpath = join(mesh_root, person, "%06d.obj" % frame)
    uv_fpath = join(data_root, "uv_table.npy")
    my_mesh = trimesh.load(mesh_fpath)
    print(my_mesh.vertices.shape, my_mesh.faces.shape)
    V = my_mesh.vertices
    F = my_mesh.faces
    T = np.load(uv_fpath)

    proj = TextureProjector3d(T, F=F)
    n_pts = 10000
    height, width = 1024, 1024
    pts = proj.random_sample(n_pts)
    pts3d = proj.query3d(pts, V)

    vert_color = torch.rand(n_pts, 3)
    vert_radii = torch.full((n_pts, 1), 0.01)
    vert_pos = torch.from_numpy(pts3d[None, ...]).type(torch.float32)
    vert_col = vert_color[None, ...]
    vert_rad = vert_radii[None, ...]
    pulsar = PulsarLayer(n_pts, height, width)
    img = pulsar(vert_pos, vert_col, vert_rad, rvec, Cs, K)
    img = torch.squeeze(img)
    print("pulsar gen image shape: ", img.shape)
    save_path = "/home/user/output"
    save_fpath = join(
        save_path, person + "_" + str(num_view) + "_" + str(frame) + "_new2.png"
    )
    imageio.imsave(save_fpath, (img.cpu().detach() * 255.0).to(torch.uint8).numpy())

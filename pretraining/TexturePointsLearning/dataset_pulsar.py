import sys
sys.path.insert(0,"")
import os 
from os.path import join
import cv2
import numpy as np 
import torch 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import ImageOps


from tqdm import tqdm 
import trimesh
import imageio

from nara.rasterizing import rasterizing
from nara.camera import Camera

from camera_pulsar import MyCamera

from utilities.nara_processing_surface_new import TextureProjector3d
from utilities.impCarv_points import PulsarLayer
from utilities.FourierFeature import input_mapping_torch

from typing import List

pid_dict={0:"313",1:"315",2:"377",3:"386",4:"387",5:"390",6:"392",7:"393",8:"394"}


class PersonPulsar(): 
    @staticmethod
    def get_all_person_ids(mesh_dir="/home/user/sample_myZjuMocap/meshes"): ## "/home/user/my_zjuMocap/meshes"
        return os.listdir(mesh_dir)
    
    def __init__(self, person_id:int, mesh_number:List, 
                 dataset_type:str, root="/home/user/my_zjuMocap/"):
        
        self.pid = pid_dict[person_id]
        self.mesh_dir = join(root,"meshes")## 50meshes:"50_meshes"; allMeshes:"meshes"; SingleMesh:"myMesh"        
        self.image_dir = join(root,"images")
        self.mask_dir = join(root,"masks")
        self.smpl_dir = join(root,"smpl_params",f"{self.pid}_smpl_params")
        self.person_path = join(self.mesh_dir,self.pid) ## for all meshes
        # self.person_path = self.mesh_dir ## for customize number of meshes
        
        ######    Enable selecting mesh   ########
        # self.frames = sorted(os.listdir(self.person_path))
        frames = sorted(os.listdir(self.person_path))
        # print(f"frames: {type(frames)},{frames}")
        self.frames = [frames[i] for i in mesh_number]
        print(f"self.frames: {type(self.frames)},{self.frames}")
        ##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$##    
        assert(len(self.frames)>0)
        self.cam_path = join(root,"cameras",self.pid)
        self.cameras = MyCamera.load_cam_data(self.cam_path)
        
        print(f"Dataset for: {dataset_type}")
        if (dataset_type=="fit" or dataset_type=="test"):
            self.cid_all = list(self.cameras.keys())
        elif(dataset_type=="validate"):
            self.cid_all=[1]     
        # self.cid_all =[1,2,4,7,13,18]
        print("cameras: ",self.cid_all)
        self.frames_per_camera={}
    
    def get_num_frames(self):
        return len(os.listdir(self.person_path))

    def get_total_frames(self):
        return len(self.get_cid_frame_pairs())

    def get_frames(self):
        return self.frames
    
    def get_frames_for_camera(self,cid):
        if cid not in self.frames_per_camera:
            valid_frames = []
            for frame in self.frames: 
                valid_frames.append(frame)
            self.frames_per_camera[cid]= valid_frames
        return self.frames_per_camera
    
    def get_cid_frame_pairs(self):
        self._enumerate_cid_frame_pairs= []
        for cid in sorted(self.cid_all):
            frames_per_camera = self.get_frames_for_camera(cid)
            for frame in frames_per_camera[cid]:
                self._enumerate_cid_frame_pairs.append((cid,frame))
        
        return self._enumerate_cid_frame_pairs


class PersonDatasetPulsar(Dataset):
    def __init__(self,person,sampling_points=300000,
                 transform=None, latent_dim=3):
        self.person = person 
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.uv_fpath = "/home/user/my_zjuMocap/uv_table.npy"
        self.points = sampling_points
        self.transformation = transform
        self.latent_dim = latent_dim
    def __len__(self):
        return self.person.get_total_frames()

    def __getitem__(self,idx):
        pid = self.person.pid 
        frame,cid = self.get_frame_and_cid(idx)
        mesh_fpath = join(self.person.mesh_dir,pid,frame) ### For all the meshes
        my_mesh = trimesh.load(mesh_fpath)
        V = my_mesh.vertices
        F = my_mesh.faces
        T = np.load(self.uv_fpath)
        
        proj = TextureProjector3d(T,F=F)
        pts3d,pts_uv,normal_noise,noramls = proj.random_sample(self.points,V)
        pts_3d = pts3d.astype(np.float32)
        gamma_pts3d = input_mapping_torch(torch.from_numpy(pts_3d), self.device).type(torch.float64)
        # print("pts_uv: ",np.max(pts_uv),np.min(pts_uv))
        pts_uv = 2.*(pts_uv - np.min(pts_uv))/np.ptp(pts_uv)-1
        # print("pts_uv: ",np.max(pts_uv),np.min(pts_uv))
        

        smpl_fpath = join(self.person.smpl_dir,frame).replace("obj","npy")
        smpl_params = np.load(smpl_fpath, allow_pickle= True)[()]
        #### dict_keys(['poses', 'Rh', 'Th', 'shapes'])        
        smpl_poses = smpl_params["poses"]
        smpl_poses = torch.from_numpy(smpl_poses)

        
        img_path = join(self.person.image_dir,pid,str(cid),frame.replace("obj","jpg"))
        mask_path = join(self.person.mask_dir,pid,str(cid),frame.replace("obj","png"))
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path,0)
        mask_tensor =  torch.tensor(mask, dtype=torch.float32)
        mask_tensor = torch.unsqueeze(mask_tensor,0)

        if self.transformation: 
            # uv_img = self.transformation(uv_img)
            # normal_img=self.transformation(normal_img)
            img = self.transformation(img)
            # mask = self.transformation(mask)
        
        

        # latent_img_dir = "/home/user/model_imp_carv/learning_color/latent_images"
        # epoch = 80
        # latent_img_fpath = join(latent_img_dir, "ep%06d" % epoch, "lt_img_313.pth")
        # latent_image = torch.load(latent_img_fpath).squeeze(0)

        # latent_dir = "/home/user/model_imp_carv/learning_color/latent_codes"
        latent_dir = f"/home/user/LearnTexturePoints_and_LatentCodes/Person_{pid}/SinglePose/LatentCodes_{self.latent_dim}Ch/latent_codes"
        epoch = 80
        latent_fpath = join(latent_dir, "ep%06d" % epoch, f"latent_{pid}.pth")
        latent = torch.load(latent_fpath).squeeze(0)

        # vert_col = torch.rand(self.points,3) ### Don't need latent image will be used instead of this
        # num_channels= 32
        vert_radii = torch.full((self.points,1),0.01)
        ##### ORIGINAL #####
        my_cam = MyCamera(self.person.cameras,cid)
        K,rvec,Cs = my_cam.cam_for_pulsar()
        R = my_cam.R
        RT = my_cam.RT
        #### Normalized Direction to the Camera
        ## ``` ndir = (cam.C - pt3d)/||cam.C - pt3d|| ```
        denominator = Cs - pts3d 
        # print(f"Denominator:{denominator.shape}")
        norm_dir2cam = ((Cs- pts3d) / denominator).type(torch.float32)
        # print(f"norm dir2cam: {norm_dir2cam.shape}")
        fft_normDirection = input_mapping_torch(norm_dir2cam, self.device) ### painter_net
        ###########################################
        return {"pts3d": pts3d, "pts_uv": pts_uv, "normal_noise": normal_noise, 
            "pid": pid, "image": img, "mask": mask_tensor, "normals": noramls, 
            "rvec": rvec, "Cs": Cs,"Ks": K, "vert_radii": vert_radii,
            "smpl_poses": smpl_poses, "gamma_pts3d":gamma_pts3d,
            "latent": latent, "fft_normDirection": fft_normDirection,
            "cid":cid, "R":R, "RT": RT }
               

    def get_frame_and_cid(self,idx):
        cid,frame = self.person.get_cid_frame_pairs()[idx]
        return frame,cid 

#### USE   RP to run ``` rp run --script="my_run.sh"    ```

if __name__ =="__main__":
    from numpy.random import default_rng
    n_pts = 10000 ## Number of points for sampling
    height,width = 1024,1024 ## H,W for pulsar images
    ######pid_dict={0:"313",1:"315",2:"377",3:"386",4:"387",5:"390",6:"392",7:"393",8:"394"}

    person_id = 0
    mesh_num = [0] 
    dataset_type = "fit"
    person_1 = PersonPulsar(person_id, mesh_num, dataset_type)
    print("person_1.pid: ",person_1.pid) ## person_1.pid:  313
    print("Number of frames per camera: ",person_1.get_num_frames()) ## person_1.get_num_frames():  1470
    print("total number of frames: ", person_1.get_total_frames()) ## total:  num_cam x frames_per_cam
    person1_dataset = PersonDatasetPulsar(person_1,n_pts)
    batch_size = 4
    person1_dataloader = DataLoader(person1_dataset,batch_size)   
    
    """
    # for data in tqdm(person1_dataloader,desc="data"):
    #     # print("data: ",data.keys())
    #     #### data:  dict_keys(['vertices', 'faces', 'texture', 'rvec', 'Cs', 'Ks', 'vert_color', 'vert_radii'])
    #     # V,F,T = data["vertices"],data["faces"],data["texture"] ## V:  torch.Size([4, 6890, 3]) F:  torch.Size([4, 13776, 3]) T:  torch.Size([4, 6890, 2])
    #     pts3d,pts_uv, normal_noise = data["pts3d"],data["pts_uv"],data["normal_noise"] ## torch.Size([4, 10000, 3]) torch.Size([4, 10000, 2]) torch.Size([4, 10000, 1])
    #     pid, imgs, masks = data["pid"], data["image"],data["mask"] ####  Image : torch.Size([4, 1024, 1024, 3]) Mask:  torch.Size([4, 1024, 1024])
    #     # print("Image :", imgs.shape,"Mask: ", masks.shape)
    #     rvec,Cs,Ks  = data["rvec"],data["Cs"],data["Ks"] ## rvec:  torch.Size([4, 3]) Cs:  torch.Size([4, 3]) Ks:  torch.Size([4, 3, 3])
    #     vert_radii =data["vert_radii"] ## v_col:  torch.Size([4, 10000, 3]) v_rad:  torch.Size([4, 10000, 1])
    #     norm_dir2cam = data["norm_dir2cam"]
    #     print(f"norm dir2cam: {norm_dir2cam.shape}")
    #     normals = data["normals"] ###  (batch_size x n_pts x 3)
    #     normal_vec = normals * normal_noise
    #     print("normals.shape: ", normals.shape)
    #     print("normal_vec.shape:" , normal_vec.shape)
    #     # print("normals.shape: ", normals.shape,np.min(normals,axis=0),np.max(normals,axis=0))
    #     # print("normal_vec.shape:" , normal_vec.shape, np.min(normal_vec,axis=0), np.max(normal_vec,axis=0))
    #     print("pts3d.shape: ", pts3d.shape)
    #     pts3d = pts3d + normal_vec
    #     print("pts3d.shape: ", pts3d.shape)
    #     pulsar = PulsarLayer(n_pts,height,width)
    #     # img = pulsar(pts3d.type(torch.float32),vert_col,vert_radii,rvec,Cs,Ks)
    #     # print("pulsar image shape: ", img.shape)
    #     # img = torch.squeeze(img) ### use this when batch_size=1 to save the image
    #     exit()
    """
    
    ### Check 3D Texture Points (U x V x h)
   
    for data in tqdm(person1_dataloader, desc="data"):
        pts_uv = data["pts_uv"].float()
        print(f"pts_uv: {pts_uv.shape, torch.min(pts_uv).item(), torch.max(pts_uv).item()}")
        n_samples = pts_uv.shape[1]
        rng = default_rng(12345)
        noise_fixed = rng.uniform(-0.1,0.1,(pts_uv.shape[0], n_samples,1)) ###(2, 300000, 1)
        uvw = np.concatenate((pts_uv.detach().cpu().numpy(),noise_fixed), axis=2) ###(2, 300000, 3)
        print(f"uvw : {uvw.shape, np.min(uvw), np.max(uvw)}")



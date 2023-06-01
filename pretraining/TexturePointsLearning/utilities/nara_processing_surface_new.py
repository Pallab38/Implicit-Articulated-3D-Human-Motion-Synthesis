import math as m

import numba as nb
import numpy as np
import numpy.linalg as la
import torch
from nara.rasterizing import barycentric_interpolation, edge_function
from numpy.random import default_rng


class TextureProjector3d:
    def __init__(
        self,
        T: np.ndarray,
        F: np.ndarray,
        density_estimate_range=0.02,
        min_uv_val=0.0,
        max_uv_val=1.0,
    ):
        """
        :param T: {n_vertices x 2} UV Texture
        :param F: {n_faces x 3} Faces
        :param density_estimate_range: {float} defines the range on
            the UV map which defines the region of a point
        """
        super().__init__()
        self.T = T
        self.F = F
        self.n_vertices = len(T)
        self.rng = default_rng(12345)
        n_vertices = len(T)

        self.faces_indices = list(range(len(F)))

    def random_sample(self, n_samples: int, V: np.ndarray):
        """
        :param V: {n_vertices x 3}
        """
        normal_noise = self.rng.uniform(-0.1, 0.1, (n_samples, 1))
        F = self.F
        T = self.T
        A = V[F[:, 0]]
        B = V[F[:, 1]]
        C = V[F[:, 2]]
        AB = A - B
        AC = A - C
        S = la.norm(np.cross(AB, AC), axis=1) / 2
        S = S / np.sum(S)

        faces_indices = self.rng.choice(self.faces_indices, size=n_samples, p=S)

        r1 = self.rng.uniform(size=(n_samples, 1))
        r2 = self.rng.uniform(size=(n_samples, 1))
        r1_sqrt = np.sqrt(r1)

        # random sampling on triangle as (https://www.cs.princeton.edu/~funk/tog02.pdf)
        # Random point P(equation 1 in the paper):
        # P = (1-sqrt(r1))A + (sqrt(r1)(1-r2))B + (r2 sqrt(r1))C
        # where r1,r2 ~ U[0,1]
        A = V[F[faces_indices, 0]]
        B = V[F[faces_indices, 1]]
        C = V[F[faces_indices, 2]]
        P = (1 - r1_sqrt) * A + (r1_sqrt * (1 - r2)) * B + (r2 * r1_sqrt) * C
        ## convert 'trimesh.caching.TrackedArray' to np.array
        P = np.array(P)

        normals = np.cross(B - A, C - A)  ### (100000, 3)
        # print("normals: ",normals)
        # print("normals.shape: ",normals.shape)
        arr_len = np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2 + normals[:, 2] ** 2)
        # print("arr_len: ",arr_len )
        normals[:, 0] /= arr_len
        normals[:, 1] /= arr_len
        normals[:, 2] /= arr_len
        # print("normals new: ",normals)
        # print("normals.shape: ",normals.shape)

        A_uv = T[F[faces_indices, 0]]
        B_uv = T[F[faces_indices, 1]]
        C_uv = T[F[faces_indices, 2]]
        P_uv = (
            (1 - r1_sqrt) * A_uv + (r1_sqrt * (1 - r2)) * B_uv + (r2 * r1_sqrt) * C_uv
        )

        return P, P_uv, normal_noise, normals

        # noise_fixed = self.rng.uniform(-0.1,0.1, (P_uv.shape[0], n_samples, 1))
        # return P, P_uv,normal_noise,normals, noise_fixed


if __name__ == "__main__":
    import os
    from os.path import join

    import trimesh

    person = "313"
    frame = 400
    root = "/home/user/my_zjuMocap"
    mesh_root = join(root, "meshes")
    mesh_fpath = join(mesh_root, person, "%06d.obj" % frame)
    uv_fpath = join(root, "uv_table.npy")
    print(os.path.isfile(mesh_fpath))
    mesh = trimesh.load(mesh_fpath)
    V = mesh.vertices  ## (6890, 3)
    F = mesh.faces  ## (13776, 3)
    T = np.load(uv_fpath)  ## (6890, 2)
    # print("V.shape: ", V.shape,"F.shape: ",F.shape,"T.shape: ", T.shape )
    # mesh.show()

    proj = TextureProjector3d(T, F=F)
    pts3d, pts_uv, normal_noise, normals = proj.random_sample(100000, V)
    print(
        "pts3d: ", pts3d.shape, "pts_uv: ", pts_uv.shape
    )  ## (100000, 3), ,(100000, 2)
    print(
        np.max(pts3d[:, 0]), np.min(pts3d[:, 0])
    )  ## 0.7883843925489555 -0.5527986884418117
    print(
        np.max(pts3d[:, 1]), np.min(pts3d[:, 1])
    )  ## 0.07194496434598784 -0.3398539713640302
    print(
        np.max(pts3d[:, 2]), np.min(pts3d[:, 2])
    )  ## 1.7112682269142525 0.001905703769033082
    print(
        "normal noise shape: ",
        normal_noise.shape,
        np.min(normal_noise),
        np.max(normal_noise),
    )
    #####   normal noise shape:  (100000, 1) -0.09999896464792866 0.09999884010818086

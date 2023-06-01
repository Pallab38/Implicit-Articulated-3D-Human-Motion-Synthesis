##### https://github.com/jutanke/implicit_carving/blob/master/implicit_carving/points.py
import torch
import torch.nn as nn
import pytorch3d.renderer.points.pulsar as pulsar
from typing import Optional


class PulsarLayer(nn.Module):
    def __init__(
        self,
        n_points: int,
        height: int,
        width: int,
        n_channels: int = 3,
        gamma: float = 1.0e-3,
        n_track: int = 5,
        device=None,
    ):
        super().__init__()
        self.gamma = gamma
        self.width = float(width)
        self.height = float(height)
        self.half_weight = float(width / 2)
        self.half_height = float(height / 2)
        self.renderer = pulsar.Renderer(
            width,
            height,
            n_points,
            right_handed_system=False,
            n_channels=n_channels,
            n_track=n_track,
        )
        self.bg_color = torch.zeros((n_channels,)).float().to(device)

    def forward(
        self,
        pos: torch.Tensor,
        col: torch.Tensor,
        rad: torch.Tensor,
        rvecs: torch.Tensor,
        Cs: torch.Tensor,
        Ks: torch.Tensor,
        opacity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param pos: {n_batch x n_points x 3} Vertex Position
        :param col: {n_batch x n_points x 3} Vertex Color or {[Bx]NxK tensor of channels}
        :param rad: {n_batch x n_points x 1} Vertex Radius
        :param rvecs: {n_batch x 3}
        :param Cs: {n_batch x 3}
        :param Ks: {n_batch x 3 x 3}
        """
        rad = rad[:, :, 0]
        # print("rad.shape: ", rad.shape)
        n_batch = Ks.size(0)
        Fs = Ks[:, 0, 0]
        cxs = -(Ks[:, 0, 2] - self.half_weight)
        cys = Ks[:, 1, 2] - self.half_height

        Fs = Fs.unsqueeze(1)
        cxs = cxs.unsqueeze(1)
        cys = cys.unsqueeze(1)

        znear = 0.1
        zfar = 10.0
        res_x = self.width
        res_y = self.height

        focal_length_px = Fs / res_x
        focal_length = torch.tensor(
            [
                znear - 1e-5,
            ],
            dtype=torch.float32,
            device=focal_length_px.device,
        )
        focal_length = focal_length.unsqueeze(1).repeat(n_batch, 1)
        s = focal_length / focal_length_px

        param = torch.cat([focal_length, s, cxs, cys], dim=1)

        cam_params = torch.cat([Cs, rvecs, param], dim=1)
        # print("pos:",pos.dtype,"col:",col.dtype,"rad: ",rad.dtype)
        # print("cam_params: ",cam_params.dtype, "param: ",param.dtype,"Cs:",Cs.dtype,"rvec: ", rvecs.dtype)
        # print("focal_length: ", focal_length.dtype,"s:",s.dtype,"cxs:",cxs.dtype,"cys:", cys.dtype)

        img = self.renderer(
            pos,
            col,
            rad,
            cam_params,
            # 1.0e-3,  # Renderer blending parameter gamma, in [1., 1e-5].
            gamma=self.gamma,
            max_depth=zfar,  # Maximum depth.
            min_depth=znear,
            return_forward_info=False,
            mode=0,
            bg_col=self.bg_color,
            opacity=opacity,
        )
        
        img = torch.flip(img, [1])
        # print(img.shape)
        return img
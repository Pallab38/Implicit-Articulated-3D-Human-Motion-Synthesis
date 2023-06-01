#### https://github.com/Pallab38/cvgLab_anr/blob/main/cvgLab_anr/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def down(in_ch, out_ch, with_bn=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=not with_bn),
        nn.ReLU(inplace=True),
    ]
    if with_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    else:
        layers.append(
            nn.GroupNorm(num_groups=8, num_channels=out_ch)
        )  ### num_channels should be divided by num_groups
    layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1))  ##padding=0, stride=1
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def middle(in_ch, out_ch, with_bn=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=1),  ##padding=0, stride=1
        nn.ReLU(inplace=True),
    ]
    if with_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    else:
        layers.append(
            nn.GroupNorm(num_groups=8, num_channels=out_ch)
        )  ### num_channels should be divided by num_groups
    layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1))  ##padding=0, stride=1
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def up(in_ch, out_ch, with_bn=True):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=0, stride=2),
        nn.ReLU(inplace=True),
    ]
    if with_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    else:
        layers.append(nn.GroupNorm(num_groups=8, num_channels=out_ch))
    layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=2))  ## padding=0, stride=1
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=256, with_bn=True):
        super(UNet, self).__init__()

        self.down1 = down(in_channels, mid_channels, with_bn)  ##
        self.down2 = down(mid_channels, 2 * mid_channels, with_bn)

        self.mid1 = middle(2 * mid_channels, 8 * mid_channels, with_bn)
        self.mid2 = middle(
            8 * mid_channels + 2 * mid_channels, 2 * mid_channels, with_bn
        )  ##cat[m1,d1]

        self.up1 = up(
            2 * mid_channels + 2 * mid_channels, mid_channels, with_bn
        )  ##cat[m2,d2]
        self.up2 = up(mid_channels + mid_channels, mid_channels // 2)  ##cat[u1,d1]

        self.out = nn.Conv2d(
            in_channels=mid_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
        )

    def forward(self, x):
        d1 = self.down1(x)
        # print(f"d1:{d1.shape}")
        d2 = self.down2(d1)
        # print(f"d2:{d2.shape}")
        m1 = self.mid1(d2)
        # print(f"m1:{m1.shape}")
        m2 = self.mid2(torch.cat([m1, d2], dim=1))
        # print(f"m1:{m1.shape}, cat[m1,d2]:{torch.cat([m1,d2],dim=1).shape}")
        # print(f"m2:{m2.shape}, cat[m2,d2]:{torch.cat([m2,d2], dim=1).shape}")
        u1 = self.up1(torch.cat([m2, d2], dim=1))
        # print(f"u1:{u1.shape}")
        u2 = self.up2(torch.cat([u1, d1], dim=1))
        out = self.out(u2)
        # print(f"u1:{u1.shape}, u2:{u2.shape}")
        # print(f"out.shape:{out.shape}")

        # img = torch.tanh(out[:,1:4,:,:])
        # mask = torch.sigmoid(out[:,0,:,:])
        # return img, mask

        return out


if __name__ == "__main__":
    x = torch.ones(4, 32, 1024, 1024)  #### [BxCxHxW]
    print(f"x:{x.shape}")
    # my_unet = UNet(in_channels=32, out_channels=4)
    # # print(f"my_unet Modules:{list(my_unet.modules())}")
    # img, mask = my_unet(x)
    # print(f"img:{img.shape}, mask:{mask.shape}")

import torch
from torch import nn
import torch.nn.functional as F

import nvdiffrast.torch

import gin

@gin.configurable()
class TriMipEncoding(nn.Module):
    def __init__(
        self,
        n_levels: int,
        plane_size: int,
        feature_dim: int,
        include_xyz: bool = False,
        include_cp: bool = False,
        comb: int = 5,
        feature_dim_factor: int = 1,
    ):
        super(TriMipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_size = plane_size
        self.feature_dim = feature_dim
        self.feature_dim_factor = feature_dim_factor
        self.include_xyz = include_xyz
        self.include_cp = include_cp
        self.comb = comb
        scale=0.2

        if not self.include_cp:
            self.register_parameter(
                "fm",
                nn.Parameter(torch.zeros(3, plane_size, plane_size, feature_dim)),
            )
            self.init_parameters()
            self.dim_out = (
                self.feature_dim * 3 + 3 if include_xyz else self.feature_dim * 3
            )
            self.f_l = 3
        else:
            # load from cpk
            #self.fm = torch.load("./salmon/fd1/nerf_synthetic/salmon_f1/Tri-MipRF/model_w_ds/fm.ckpt")
            #self.fm = torch.load("./log_sample/fd1/nerf_synthetic/coffee_f1/Tri-MipRF/model_w_ds/fm.ckpt")
            self.fm = torch.load("./test/nerf_synthetic/coffee_f1/Tri-MipRF/model_w_ds/fm.ckpt")

            #checkpoint = torch.load("./test/nerf_synthetic/coffee_f1/Tri-MipRF/model_w_ds/fm.ckpt", map_location='cpu')
            #checkpoint = checkpoint.contiguous()  # Make contiguous!
            #self.register_buffer("fm", checkpoint.clone())
            #print(f"Checkpoint registered: shape={self.fm.shape}, "f"contiguous={self.fm.is_contiguous()}, dtype={self.fm.dtype}")

            self.register_parameter(
                "x",
                nn.Parameter(scale * torch.randn((1, self.feature_dim * self.feature_dim_factor, self.plane_size, 1)))
                )
            self.register_parameter(
                "y",
                nn.Parameter(scale * torch.randn((1, self.feature_dim * self.feature_dim_factor, self.plane_size, 1)))
                )
            self.register_parameter(
                "z",
                nn.Parameter(scale * torch.randn((1, self.feature_dim * self.feature_dim_factor, self.plane_size, 1)))
                )
            
            #print("=====> loading self.fm, x: ", self.x.shape) #[1, 16,512,1]

            if self.comb == 1 or self.comb == 2:
                self.f_l = 3
            elif self.comb == 3:
                self.f_l = 4
            elif self.comb == 4:
                self.f_l = 6
            elif self.comb == 5:
                self.f_l = 3 + self.feature_dim_factor
            elif self.comb == 6:
                self.f_l = 1

            self.dim_out = (
                self.feature_dim * self.f_l + 3 if include_xyz else self.feature_dim * self.f_l
            )

    def init_parameters(self) -> None:
        # Important for performance
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def forward(self, x, level):
        # x in [0,1], level in [0,max_level]
        # x is Nx3, level is Nx1
        #print("===> shape", x.shape, level.shape)
        if 0 == x.shape[0]:
            return torch.zeros([x.shape[0], self.feature_dim * self.f_l]).to(x)
        decomposed_x = torch.stack(
            [
                x[:, None, [1, 2]],
                x[:, None, [0, 2]],
                x[:, None, [0, 1]],
            ],
            dim=0,
        )  # 3xNx1x2

        if 0 == self.n_levels:
            level = None
        else:
            # assert level.shape[0] > 0, [level.shape, x.shape]
            # torch.stack([level, level, level], dim=0)
            level = torch.broadcast_to(
                level, decomposed_x.shape[:3]
            ).contiguous()

        def extract_plane_features():
            
            fm_to_use = self.fm
            if not fm_to_use.is_contiguous():
                print("WARNING: fm not contiguous, fixing...")
                fm_to_use = fm_to_use.contiguous()

            enc = nvdiffrast.torch.texture(
                fm_to_use,
                decomposed_x,
                #mip_level_bias=level,
                boundary_mode="clamp",
                max_mip_level= 0, #self.n_levels - 1,
            )  # 3xNx1xC
            
            return enc

        if not self.include_cp:
            # extract triplane features
            enc = extract_plane_features()
            enc = (
                enc.permute(1, 2, 0, 3)
                .contiguous()
                .view(
                    x.shape[0],
                    self.feature_dim * 3,
                )
            )  # Nx(3C)

        else:
            # extract triplane features
            #print("extract plane here", x.shape)
            with torch.no_grad():
                enc = extract_plane_features()
            # for extracting cp vector features
            decomposed_x_ = torch.stack(
                [
                    x[:, None, [0, 0]],
                    x[:, None, [1, 1]],
                    x[:, None, [2, 2]],
                ],
                dim=0,
            )  # 3xNx1x1
            #print(decomposed_x_.shape)
            x_input = decomposed_x_[0].unsqueeze(0)
            x_cp_enc = F.grid_sample(self.x, x_input, align_corners=True).view(-1, x_input.shape[1])
            y_input = decomposed_x_[1].unsqueeze(0)
            y_cp_enc = F.grid_sample(self.y, y_input, align_corners=True).view(-1, y_input.shape[1])
            z_input = decomposed_x_[2].unsqueeze(0)
            z_cp_enc = F.grid_sample(self.z, z_input, align_corners=True).view(-1, z_input.shape[1])
            # CxN
            x_cp_enc = torch.transpose(x_cp_enc, 0, 1)
            y_cp_enc = torch.transpose(y_cp_enc, 0, 1)
            z_cp_enc = torch.transpose(z_cp_enc, 0, 1)

            yz_enc = enc[0].squeeze(1) # NxC
            xz_enc = enc[1].squeeze(1)
            xy_enc = enc[2].squeeze(1)

            ### How to combine those features? concate, product, add?
            if self.comb == 1:
                yz = yz_enc + x_cp_enc
                xz = xz_enc + y_cp_enc
                xy = xy_enc + z_cp_enc
                enc = torch.stack([yz,xz,xy],dim=0) # 3xNxC
                enc = (enc.permute(1,0,2).contiguous().view(x.shape[0], self.feature_dim*self.f_l))

            elif self.comb == 2:
                yz = yz_enc * x_cp_enc
                xz = xz_enc * y_cp_enc
                xy = xy_enc * z_cp_enc
                enc = torch.stack([yz,xz,xy],dim=0) # 3xNxC
                enc = (enc.permute(1,0,2).contiguous().view(x.shape[0], self.feature_dim*self.f_l))

            elif self.comb == 3:
                cp = x_cp_enc * y_cp_enc * z_cp_enc
                enc = torch.stack([yz_enc,xz_enc,xy_enc, cp],dim=0) # 4xNxC
                enc = (enc.permute(1,0,2).contiguous().view(x.shape[0], self.feature_dim*self.f_l))

            elif self.comb == 4:
                enc = torch.stack([yz_enc,xz_enc,xy_enc, x_cp_enc, y_cp_enc, z_cp_enc],dim=0) # 6xNxC
                enc = (enc.permute(1,0,2).contiguous().view(x.shape[0], self.feature_dim*self.f_l))

            elif self.comb == 5:
                cp = x_cp_enc * y_cp_enc * z_cp_enc
                enc = torch.cat((yz_enc,xz_enc,xy_enc, cp), dim = -1)
                #print(cp.shape, enc.shape)

            elif self.comb == 6:
                plane_enc = torch.sum(enc, dim=0).squeeze(1).contiguous().view(x.shape[0], self.feature_dim)
                cp = x_cp_enc * y_cp_enc * z_cp_enc
                enc = (plane_enc * cp).contiguous()

        if self.include_xyz:
            enc = torch.cat([x, enc], dim=-1)
        return enc

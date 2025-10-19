import math
from typing import Union, List, Dict,Tuple

import gin
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure as ssim

from utils.ray import RayBundle
from utils.render_buffer import RenderBuffer


# @gin.configurable()
class RFModel(nn.Module):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        samples_per_ray: int = 1024,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.samples_per_ray = samples_per_ray
        self.render_step_size = (
            (self.aabb[3:] - self.aabb[:3]).max()
            * math.sqrt(3)
            / samples_per_ray
        ).item()
        aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
        self.aabb_size = aabb_max - aabb_min
        assert (
            self.aabb_size[0] == self.aabb_size[1] == self.aabb_size[2]
        ), "Current implementation only supports cube aabb"
        self.field = None
        self.ray_sampler = None

    def contraction(self, x):
        aabb_min, aabb_max = self.aabb[:3].unsqueeze(0), self.aabb[
            3:
        ].unsqueeze(0)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        return x

    def before_iter(self, step):
        pass

    def after_iter(self, step):
        pass

    def forward(
        self,
        rays: RayBundle,
        background_color=None,
    ):
        raise NotImplementedError

    @gin.configurable()
    def get_optimizer(
        self, lr=1e-3, weight_decay=1e-5, feature_lr_scale=10.0, **kwargs
    ):
        raise NotImplementedError

    @gin.configurable()
    def compute_loss(
        self,
        rays: RayBundle,
        rb: RenderBuffer,
        target: RenderBuffer,
        # Configurable
        metric='smooth_l1',
        **kwargs
    ) -> Dict:
        if 'smooth_l1' == metric:
            loss_fn = F.smooth_l1_loss
        elif 'mse' == metric:
            loss_fn = F.mse_loss
        elif 'mae' == metric:
            loss_fn = F.l1_loss
        else:
            raise NotImplementedError

        alive_ray_mask = (rb.alpha.squeeze(-1) > 0).detach()
        loss = loss_fn(
            rb.rgb[alive_ray_mask], target.rgb[alive_ray_mask], reduction='none'
        )
        loss = (
            loss * target.loss_multi[alive_ray_mask]
        ).sum() / target.loss_multi[alive_ray_mask].sum()
        return {'total_loss': loss}
    
    @staticmethod
    def infer_hw_from_rb(rb) -> Tuple[Union[int, None], Union[int, None]]:
        H = getattr(rb, "height", None)
        W = getattr(rb, "width", None)
        return H, W
    
    @staticmethod
    def to_nchw_image(x: torch.Tensor, hw: Union[Tuple[int, int], None] = None):
        if x is None:
            return None

        # already 4D
        if x.ndim == 4:
            # (N,H,W,C) -> (N,C,H,W)
            if x.shape[-1] in (1, 3):
                return x.permute(0, 3, 1, 2).contiguous()
            # (N,C,H,W)
            if x.shape[1] in (1, 3):
                return x.contiguous()
            return None

        # 3D cases
        if x.ndim == 3:
            # (H,W,C) -> (1,C,H,W)
            if x.shape[-1] in (1, 3):
                return x.permute(2, 0, 1).unsqueeze(0).contiguous()
            # (C,H,W) -> (1,C,H,W)
            if x.shape[0] in (1, 3):
                return x.unsqueeze(0).contiguous()
            # (N, H*W, 3) with hw
            if x.shape[-1] == 3 and hw is not None:
                H, W = hw
                N = x.shape[0]
                return x.view(N, H, W, 3).permute(0, 3, 1, 2).contiguous()
            return None

        # 2D rays: (H*W, 3) with hw
        if x.ndim == 2 and x.shape[-1] in (1, 3) and hw is not None:
            H, W = hw
            return x.view(H, W, x.shape[-1]).permute(2, 0, 1).unsqueeze(0).contiguous()

        return None
    
    @staticmethod
    def infer_data_range(x: torch.Tensor) -> float:
        mx = float(x.max().detach()); mn = float(x.min().detach())
        rng = mx - mn
        if rng <= 0:
            return 1.0
        return 255.0 if mx > 1.2 else 1.0
   
    @gin.configurable()
    def compute_metrics(
        self,
        rays: RayBundle,
        rb: RenderBuffer,
        target: RenderBuffer,

        hw: Union[Tuple[int, int], None] = None, 
        # Configurable
        **kwargs
    ) -> Dict:
        # ray info
        alive_ray_mask = (rb.alpha.squeeze(-1) > 0).detach()
        rendering_samples_actual = rb.num_samples[0].item()
        ray_info = {
            'num_alive_ray': alive_ray_mask.long().sum().item(),
            'rendering_samples_actual': rendering_samples_actual,
            'num_rays': len(target),
        }
        # quality
        #quality = {'PSNR': peak_signal_noise_ratio(rb.rgb, target.rgb).item()}
        psnr_val = peak_signal_noise_ratio(rb.rgb, target.rgb).item()

        
        inferred_hw = hw or self.infer_hw_from_rb(rb)
        if inferred_hw == (None, None):
            inferred_hw = None

        # reshape to (N,C,H,W) if possible
        pred_nchw = self.to_nchw_image(rb.rgb, inferred_hw)   # use inferred_hw
        tgt_nchw  = self.to_nchw_image(target.rgb, inferred_hw)

        # SSIM if we have image layout
        if pred_nchw is not None and tgt_nchw is not None:
            pred_nchw = pred_nchw.float()
            tgt_nchw  = tgt_nchw.float()
            data_range = self.infer_data_range(tgt_nchw)
            ssim_val = ssim(pred_nchw, tgt_nchw, data_range=data_range).item()
        else:
            ssim_val = float('nan')
        
        quality = {'PSNR': psnr_val, 'SSIM': ssim_val}
        return {**ray_info, **quality}

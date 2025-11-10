# python tokenizer/losses.py
# tokenizer/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from einops import rearrange

class MSELoss(nn.Module):
    """
    Pixel-wise Mean Squared Error loss, supports optional mask.
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            recon: Tensor of shape (B, C, H, W) or (B, T, N, D) for patches
            target: Tensor of same shape
            mask: Optional tensor broadcastable to input shape
        Returns:
            scalar Tensor (loss)
        """
        if mask is not None:
            # ensure mask broadcast shape
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B,1,H,W) -> (B,1,H,W)
            # compute per-pixel squared error
            loss = (recon - target).pow(2)
            loss = loss * mask
            denom = mask.sum().clamp(min=1.0)
            return loss.sum() / denom
        else:
            return F.mse_loss(recon, target)

class CombinedLoss(nn.Module):
    def __init__(self, lpips_weight=0.1, lpips_net='alex', patch_size=None, frame_size=None):
        super().__init__()
        self.mse_loss = MSELoss()
        # LPIPS expects input in [-1, 1] range
        self.lpips_fn = lpips.LPIPS(net=lpips_net).eval()
        self.lpips_weight = lpips_weight
        
        # For unpatchifying
        self.patch_size = patch_size
        self.frame_size = frame_size
        if patch_size is not None:
            from tokenizer.patchify_mask import Patchifier
            self.patchifier = Patchifier(patch_size=patch_size)
        
        # Freeze LPIPS network
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
    
    def unpatchify_batch(self, patches: torch.Tensor) -> torch.Tensor:  # FIXED INDENTATION
        """Direct einops unpatchify - FIXED VERSION"""
        B, T, N, D = patches.shape
        H, W = self.frame_size
        ps = self.patch_size
        C = 3
        num_h = H // ps
        num_w = W // ps
        
        images = rearrange(
            patches,
            "b t (nh nw) (c ph pw) -> b t c (nh ph) (nw pw)",
            nh=num_h, nw=num_w, c=C, ph=ps, pw=ps
        )
        
        return images

    def forward(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            recon: (B, T, N, D) patches
            target: (B, T, N, D) patches
            mask: Optional (B, T, N) mask
        """
        # 1. Compute MSE loss on patches
        mse_loss = self.mse_loss(recon, target, mask)
        
        # 2. Ensure patches are in valid [0, 1] range BEFORE unpatchifying
        recon = torch.sigmoid(recon) if recon.min() < 0 or recon.max() > 1 else recon
        target = torch.clamp(target, 0, 1)  # Ensure target is valid too
        
        # 3. Convert patches to images for LPIPS
        recon_images = self.unpatchify_batch(recon)  # (B, T, C, H, W)
        target_images = self.unpatchify_batch(target)  # (B, T, C, H, W)
        
        # 4. Reshape from (B, T, C, H, W) to (B*T, C, H, W) for LPIPS
        B, T, C, H, W = recon_images.shape
        recon_flat = recon_images.reshape(B * T, C, H, W)
        target_flat = target_images.reshape(B * T, C, H, W)
        
        # 5. Normalize to [-1, 1] for LPIPS (no clamping needed now)
        recon_normalized = recon_flat * 2 - 1
        target_normalized = target_flat * 2 - 1
        
        # 6. Compute LPIPS
        lpips_values = self.lpips_fn(recon_normalized, target_normalized)
        lpips_value = lpips_values.mean()
        
        # 7. Combined loss
        total_loss = mse_loss + self.lpips_weight * lpips_value
        
        return total_loss, {
            'mse': mse_loss.item(),
            'lpips': lpips_value.item(),
            'total': total_loss.item()
        }
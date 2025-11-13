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

class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) loss.
    Expects inputs in range [-1, 1].
    """
    def __init__(self, lpips_net='alex', patch_size=None, frame_size=None):
        super(LPIPSLoss, self).__init__()
        self.lpips_fn = lpips.LPIPS(net=lpips_net).eval()
        
        # For unpatchifying
        self.patch_size = patch_size
        self.frame_size = frame_size
        if patch_size is not None:
            from tokenizer.patchify_mask import Patchifier
            self.patchifier = Patchifier(patch_size=patch_size)
        
        # Freeze LPIPS network
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
    
    def unpatchify_batch(self, patches: torch.Tensor) -> torch.Tensor:
        """Convert (B, T, N, D) to (B, T, C, H, W) using Patchifier"""
        B, T, N, D = patches.shape
        
        # Process each frame
        all_frames = []
        for b in range(B):
            batch_frames = []
            for t in range(T):
                # Unpatchify single frame (1, N, D) -> (1, C, H, W)
                frame = self.patchifier.unpatchify(
                    patches[b:b+1, t], 
                    frame_size=self.frame_size,
                    patch_size=self.patch_size
                )  # (1, C, H, W)
                batch_frames.append(frame[0])  # (C, H, W)
            all_frames.append(torch.stack(batch_frames))  # (T, C, H, W)
        
        images = torch.stack(all_frames)  # (B, T, C, H, W)
        return images

    def forward(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None): 
        # 1. Convert patches to images (both should already be in [0,1] from model)
        recon_images = self.unpatchify_batch(recon)  # (B, T, C, H, W)
        target_images = self.unpatchify_batch(target)  # (B, T, C, H, W)
        
        # 2. Reshape and normalize for LPIPS
        B, T, C, H, W = recon_images.shape
        recon_flat = recon_images.reshape(B * T, C, H, W)
        target_flat = target_images.reshape(B * T, C, H, W)
        
        # Normalize to [-1, 1] for LPIPS
        recon_normalized = (recon_flat * 2 - 1).clamp(-1, 1)
        target_normalized = (target_flat * 2 - 1).clamp(-1, 1)
        
        # 3. Compute LPIPS
        lpips_values = self.lpips_fn(recon_normalized, target_normalized)
        lpips_loss = lpips_values.mean()
        
        return lpips_loss

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
    
    def unpatchify_batch(self, patches: torch.Tensor) -> torch.Tensor:
        """Convert (B, T, N, D) to (B, T, C, H, W) using Patchifier"""
        B, T, N, D = patches.shape
        
        # Process each frame
        all_frames = []
        for b in range(B):
            batch_frames = []
            for t in range(T):
                # Unpatchify single frame (1, N, D) -> (1, C, H, W)
                frame = self.patchifier.unpatchify(
                    patches[b:b+1, t], 
                    frame_size=self.frame_size,
                    patch_size=self.patch_size
                )  # (1, C, H, W)
                batch_frames.append(frame[0])  # (C, H, W)
            all_frames.append(torch.stack(batch_frames))  # (T, C, H, W)
        
        images = torch.stack(all_frames)  # (B, T, C, H, W)
        return images

    def forward(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        # 1. Compute MSE loss on patches
        mse_loss = self.mse_loss(recon, target, mask)
        
        # 2. Convert patches to images (both should already be in [0,1] from model)
        recon_images = self.unpatchify_batch(recon)  # (B, T, C, H, W)
        target_images = self.unpatchify_batch(target)  # (B, T, C, H, W)
        
        # 3. Reshape and normalize for LPIPS
        B, T, C, H, W = recon_images.shape
        recon_flat = recon_images.reshape(B * T, C, H, W)
        target_flat = target_images.reshape(B * T, C, H, W)
        
        # Normalize to [-1, 1] for LPIPS
        recon_normalized = (recon_flat * 2 - 1).clamp(-1, 1)
        target_normalized = (target_flat * 2 - 1).clamp(-1, 1)
        
        # 4. Compute LPIPS
        lpips_values = self.lpips_fn(recon_normalized, target_normalized)
        lpips_value = lpips_values.mean()
        
        # 5. Combined loss
        total_loss = mse_loss + self.lpips_weight * lpips_value
        
        return total_loss, {
            'mse': mse_loss.item(),
            'lpips': lpips_value.item(),
            'total': total_loss.item()
        }
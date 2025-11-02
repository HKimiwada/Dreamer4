import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMSELoss(nn.Module):
    """Pixel-wise MSE with optional mask."""
    def __init__(self):
        super().__init__()

    def forward(self, recon, target, mask=None):
        if mask is not None:
            mse = (recon - target) ** 2
            mse = mse * mask
            return mse.sum() / mask.sum().clamp(min=1)
        else:
            return F.mse_loss(recon, target)


class PerceptualLoss(nn.Module):
    """Optional perceptual or patch-based loss."""
    def __init__(self, patch_h=16, patch_w=16):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, recon, target, mask=None):
        B, C, H, W = recon.shape
        # Make sure patches divide evenly
        assert H % self.patch_h == 0 and W % self.patch_w == 0, \
            f"Image size {(H, W)} not divisible by patch size {(self.patch_h, self.patch_w)}"
        
        if mask is not None:
            # Broadcast mask to 3 channels if needed
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            recon = recon * mask
            target = target * mask
        
        # Flatten patches
        recon_patches = recon.unfold(2, self.patch_h, self.patch_h).unfold(3, self.patch_w, self.patch_w)
        target_patches = target.unfold(2, self.patch_h, self.patch_h).unfold(3, self.patch_w, self.patch_w)
        
        # Compute MSE per patch
        loss = F.mse_loss(recon_patches, target_patches)
        return loss


class CombinedLoss(nn.Module):
    """Weighted sum of masked MSE and perceptual loss."""
    def __init__(self, alpha=0.5, patch_h=16, patch_w=16):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = MaskedMSELoss()
        self.perc_loss = PerceptualLoss(patch_h, patch_w)

    def forward(self, recon, target, mask=None):
        mse_val = self.mse_loss(recon, target, mask)
        perc_val = self.perc_loss(recon, target, mask)
        total = (1 - self.alpha) * mse_val + self.alpha * perc_val
        return total, {'mse': mse_val.item(), 'perc': perc_val.item()}


# --- Optional quick test ---
if __name__ == "__main__":
    recon = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)
    mask = torch.ones_like(recon[:, :1, :, :])
    loss_fn = CombinedLoss(alpha=0.3)
    total, parts = loss_fn(recon, target, mask)
    print("Total:", total.item(), parts)

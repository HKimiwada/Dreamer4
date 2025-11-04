# python tokenizer/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    # You must install the `lpips` package (e.g. pip install lpips) to use LPIPS loss.

class MSELoss(nn.Module):
    """
    Pixel-wise Mean Squared Error loss, supports optional mask.
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            recon: Tensor of shape (B, C, H, W)
            target: Tensor of same shape
            mask: Optional tensor broadcastable to (B, C, H, W) or (B, 1, H, W)
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
    LPIPS loss (Learned Perceptual Image Patch Similarity).
    Requires the `lpips` package.
    Inputs must be in the proper range (by default: [-1, +1] for LPIPS).
    """
    def __init__(self, net: str = 'vgg', version: str = '0.1', normalize_inputs: bool = True):
        """
        Args:
            net: which backbone (‘alex’, ‘vgg’, etc) for LPIPS. Refer to package docs. :contentReference[oaicite:1]{index=1}
            version: version of the LPIPS weights.
            normalize_inputs: if True, expects input in [0,1] and will convert to [-1,1] internally.
        """
        super(LPIPSLoss, self).__init__()
        if not _LPIPS_AVAILABLE:
            raise ImportError("lpips package not found. Install via `pip install lpips` to use LPIPSLoss.")
        self.lpips_fn = lpips.LPIPS(net=net, version=version).eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False
        self.normalize_inputs = normalize_inputs

    def forward(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            recon, target: tensors shape (B, 3, H, W)
            mask: optional mask (B,1,H,W) or (B,3,H,W) — will zero out masked regions.
        Returns:
            scalar Tensor (loss)
        """
        # normalize to [-1, +1] if needed
        if self.normalize_inputs:
            recon_norm = recon * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
        else:
            recon_norm = recon
            target_norm = target

        # apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # broadcast mask to 3 channels if necessary
            if mask.shape[1] == 1 and recon_norm.shape[1] == 3:
                mask = mask.repeat(1, 3, 1, 1)
            recon_norm = recon_norm * mask
            target_norm = target_norm * mask

        # compute lpips distance
        loss_val = self.lpips_fn(recon_norm, target_norm)
        # Typically returns shape (B,1,1,1) or (B,), so mean over batch
        return loss_val.mean()

class CombinedLoss(nn.Module):
    """
    Wrapper loss combining MSE + LPIPS with weighting.
    total_loss = (1 - alpha) * mse + alpha * lpips
    """
    def __init__(self, alpha: float = 0.5, lpips_net: str = 'vgg', lpips_version: str = '0.1', normalize_inputs_lpips: bool = True):
        """
        Args:
            alpha: weight for LPIPS part (between 0 and 1)
            lpips_net, lpips_version, normalize_inputs_lpips: args forwarded to LPIPSLoss
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = MSELoss()
        self.lpips_loss = LPIPSLoss(net=lpips_net, version=lpips_version, normalize_inputs=normalize_inputs_lpips)

    def forward(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Compute combined MSE + LPIPS loss.
        Works for both image (B,3,H,W) and video (B,T,3,H,W) inputs.

        Args:
            recon, target: (B, C, H, W) or (B, T, C, H, W)
            mask: optional mask matching spatial dims,
                e.g. (B, 1, H, W), (B, C, H, W), or (B, T, 1, H, W)
        Returns:
            total_loss: scalar tensor
            parts: dict with 'mse' and 'lpips'
        """
        # --- Handle video inputs (B,T,C,H,W) ---
        if recon.dim() == 5:
            B, T, C, H, W = recon.shape
            recon_2d  = recon.reshape(B * T, C, H, W)
            target_2d = target.reshape(B * T, C, H, W)
            if mask is not None:
                # (B,T,1,H,W) or (B,T,H,W) -> (B*T,1,H,W)
                if mask.dim() == 5:
                    mask_2d = mask.view(B * T, 1, H, W)
                elif mask.dim() == 4:
                    mask_2d = mask.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, 1, H, W)
                else:
                    mask_2d = None
            else:
                mask_2d = None
        else:
            # --- Handle image inputs (B,C,H,W) ---
            recon_2d, target_2d, mask_2d = recon, target, mask

        # --- Compute losses ---
        mse_val = self.mse_loss(recon, target, mask)
        lpips_val = self.lpips_loss(recon_2d, target_2d, mask_2d)

        total = (1.0 - self.alpha) * mse_val + self.alpha * lpips_val

        return total, {'mse': mse_val.item(), 'lpips': lpips_val.item()}


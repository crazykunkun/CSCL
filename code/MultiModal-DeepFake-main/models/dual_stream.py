import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class SRMConv2d(nn.Module):
    """SRM high-pass filter layer — extracts edges, texture and gradient residuals."""

    def __init__(self):
        super().__init__()
        # 3 classic 5x5 SRM kernels, each applied per-channel -> 3*3=9 output channels
        srm_kernels = self._build_srm_kernels()  # [9, 3, 5, 5]
        self.register_buffer("weight", srm_kernels)

    @staticmethod
    def _build_srm_kernels():
        # --- kernel 1: 1st-order edge ---
        k1 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=np.float32)

        # --- kernel 2: 2nd-order texture ---
        k2 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  2, -4,  2,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=np.float32) / 4.0

        # --- kernel 3: 3rd-order gradient ---
        k3 = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1],
        ], dtype=np.float32) / 12.0

        kernels = []
        for k in [k1, k2, k3]:
            # expand single kernel to 3 input channels (one filter per channel)
            for c in range(3):
                w = np.zeros((3, 5, 5), dtype=np.float32)
                w[c] = k
                kernels.append(w)
        return torch.tensor(np.stack(kernels, axis=0))  # [9, 3, 5, 5]

    def forward(self, x):
        # x: [B, 3, H, W]
        return F.conv2d(x, self.weight, stride=1, padding=2)  # [B, 9, H, W]


class FrequencyEncoder(nn.Module):
    """Lightweight CNN backbone (ResNet-18) that keeps spatial feature maps."""

    def __init__(self, in_channels=9):
        super().__init__()
        backbone = models.resnet18(pretrained=True)

        # replace first conv to accept SRM output channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # init from pretrained weights by averaging across input channels
        with torch.no_grad():
            pretrained_w = backbone.conv1.weight  # [64, 3, 7, 7]
            self.conv1.weight.copy_(
                pretrained_w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            )

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        # avgpool and fc are intentionally dropped

    def forward(self, x):
        # x: [B, 9, H, W]  (e.g. H=W=256)
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, 128, 128]
        x = self.maxpool(x)                      # [B, 64, 64, 64]
        x = self.layer1(x)                       # [B, 64, 64, 64]
        x = self.layer2(x)                       # [B, 128, 32, 32]
        x = self.layer3(x)                       # [B, 256, 16, 16]
        x = self.layer4(x)                       # [B, 512, 8, 8]
        return x


class DualStreamImageEncoder(nn.Module):
    """Fuses ViT RGB features with SRM-based frequency features."""

    PATCH_GRID = 16  # ViT-B/16 -> 16x16 patches for 256x256 input
    VIT_DIM = 768
    FREQ_DIM = 512

    def __init__(self, rgb_encoder=None):
        super().__init__()
        self.rgb_encoder = rgb_encoder  # external ViT; pass None for standalone test
        self.srm = SRMConv2d()
        self.freq_encoder = FrequencyEncoder(in_channels=9)

        fused_dim = self.VIT_DIM + self.FREQ_DIM  # 1280
        self.projection = nn.Sequential(
            nn.Linear(fused_dim, self.VIT_DIM),
            nn.LayerNorm(self.VIT_DIM),
            nn.GELU(),
        )

    def _extract_freq_tokens(self, x):
        noise = self.srm(x)                       # [B, 9, 256, 256]
        feat = self.freq_encoder(noise)            # [B, 512, 8, 8]
        feat = F.adaptive_avg_pool2d(
            feat, (self.PATCH_GRID, self.PATCH_GRID)
        )                                          # [B, 512, 16, 16]
        feat = feat.flatten(2).transpose(1, 2)     # [B, 256, 512]
        return feat

    def forward(self, x, rgb_features=None):
        """
        Args:
            x: raw image [B, 3, 256, 256]
            rgb_features: pre-computed ViT patch tokens [B, 256, 768].
                          If None, self.rgb_encoder(x) is called.
        Returns:
            fused: [B, 256, 768]
        """
        # --- RGB stream ---
        if rgb_features is not None:
            rgb_feat = rgb_features                # [B, 256, 768]
        else:
            rgb_feat = self.rgb_encoder(x)         # [B, 256, 768]

        # --- Frequency stream ---
        freq_feat = self._extract_freq_tokens(x)   # [B, 256, 512]

        # --- Fusion ---
        fused = torch.cat([rgb_feat, freq_feat], dim=-1)  # [B, 256, 1280]
        fused = self.projection(fused)                     # [B, 256, 768]
        return fused


# --------------- verification ---------------
if __name__ == "__main__":
    B = 2

    # dummy ViT that returns [B, 256, 768]
    dummy_vit = lambda x: torch.randn(B, 256, 768)

    encoder = DualStreamImageEncoder(rgb_encoder=dummy_vit)
    x = torch.randn(B, 3, 256, 256)
    out = encoder(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    assert out.shape == (B, 256, 768), "Shape mismatch!"
    print("All checks passed.")

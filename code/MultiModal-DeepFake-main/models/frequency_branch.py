import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp


class FixedSRMConv(nn.Module):
    """固定 SRM 高通残差滤波器。

    SRM（Spatial Rich Model）常用于图像取证任务，用固定卷积核放大局部噪声残差。
    这里的卷积核不参与训练，只负责提取局部拼接、生成伪影、纹理不连续等高频异常。
    输入为灰度图 [B, 1, H, W]，输出为 3 个残差通道 [B, 3, H, W]。
    """

    def __init__(self):
        super().__init__()
        kernels = torch.tensor(
            [
                [[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 1, -2, 1, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0],
                 [0, -1, 2, -1, 0],
                 [0, 2, -4, 2, 0],
                 [0, -1, 2, -1, 0],
                 [0, 0, 0, 0, 0]],
                [[-1, 2, -2, 2, -1],
                 [2, -6, 8, -6, 2],
                 [-2, 8, -12, 8, -2],
                 [2, -6, 8, -6, 2],
                 [-1, 2, -2, 2, -1]],
            ],
            dtype=torch.float32,
        )
        kernels[0] = kernels[0] / 2.0
        kernels[1] = kernels[1] / 4.0
        kernels[2] = kernels[2] / 12.0
        # 注册为 buffer：随模型移动到 GPU，但不作为可训练参数更新。
        self.register_buffer("weight", kernels.unsqueeze(1), persistent=False)

    def forward(self, gray):
        return F.conv2d(gray, self.weight, padding=2)


class DCTHighFrequencyMap(nn.Module):
    """块 DCT 高频能量图。

    将灰度图按 block_size 切成不重叠小块，对每个块计算 DCT 系数，
    只保留 u + v >= high_frequency_start 的高频系数，并用均方能量形成 1 通道图。
    该分支主要捕获 JPEG 压缩、重采样和局部篡改造成的块频率异常。
    """

    def __init__(self, block_size=8, high_frequency_start=4):
        super().__init__()
        self.block_size = block_size
        self.high_frequency_start = high_frequency_start
        basis = self._build_dct_basis(block_size)
        mask = self._build_high_frequency_mask(block_size, high_frequency_start)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("mask", mask, persistent=False)

    @staticmethod
    def _build_dct_basis(block_size):
        # 预先构造 2D DCT 基矩阵，后续可通过矩阵乘法快速得到每个块的 DCT 系数。
        rows = []
        for u in range(block_size):
            for v in range(block_size):
                basis = []
                au = math.sqrt(1 / block_size) if u == 0 else math.sqrt(2 / block_size)
                av = math.sqrt(1 / block_size) if v == 0 else math.sqrt(2 / block_size)
                for x in range(block_size):
                    for y in range(block_size):
                        basis.append(
                            au
                            * av
                            * math.cos(math.pi * (2 * x + 1) * u / (2 * block_size))
                            * math.cos(math.pi * (2 * y + 1) * v / (2 * block_size))
                        )
                rows.append(basis)
        return torch.tensor(rows, dtype=torch.float32)

    @staticmethod
    def _build_high_frequency_mask(block_size, high_frequency_start):
        # 用 u + v 作为频率高低的简单判据，越靠右下代表频率越高。
        mask = []
        for u in range(block_size):
            for v in range(block_size):
                mask.append((u + v) >= high_frequency_start)
        return torch.tensor(mask, dtype=torch.bool)

    def forward(self, gray):
        block_size = self.block_size
        height, width = gray.shape[-2:]
        pad_h = (block_size - height % block_size) % block_size
        pad_w = (block_size - width % block_size) % block_size
        if pad_h or pad_w:
            # 尺寸不能整除 block_size 时用 reflect padding，避免边界出现硬零值。
            gray = F.pad(gray, (0, pad_w, 0, pad_h), mode="reflect")

        # unfold 后每一列是一个 block，形状约为 [B, block_size * block_size, num_blocks]。
        patches = F.unfold(gray, kernel_size=block_size, stride=block_size)
        coeffs = torch.matmul(self.basis.to(patches.device, patches.dtype), patches)
        high_coeffs = coeffs[:, self.mask, :]
        # 将多个高频系数压缩为一个能量值，得到块级高频响应。
        energy = torch.sqrt(high_coeffs.pow(2).mean(dim=1, keepdim=True) + 1e-6)
        out_h = gray.shape[-2] // block_size
        out_w = gray.shape[-1] // block_size
        return energy.view(gray.shape[0], 1, out_h, out_w)


class FFTAmplitudeHighPass(nn.Module):
    """FFT 全局高通幅度响应。

    先对灰度图做 2D FFT，再使用半径阈值构造高通掩码，仅保留高频频率成分，
    最后通过 IFFT 回到空间域并取绝对值。该分支提供全局频谱视角。
    """

    def __init__(self, cutoff=0.35):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, gray):
        # 去均值可以削弱 DC 分量，使高通响应更聚焦纹理与边缘异常。
        gray = gray - gray.mean(dim=(-2, -1), keepdim=True)
        fft = torch.fft.rfft2(gray, norm="ortho")
        height, width = gray.shape[-2:]
        fy = torch.fft.fftfreq(height, device=gray.device, dtype=gray.dtype).view(1, 1, height, 1)
        fx = torch.fft.rfftfreq(width, device=gray.device, dtype=gray.dtype).view(1, 1, 1, width // 2 + 1)
        radius = torch.sqrt(fx.pow(2) + fy.pow(2))
        # 半径大于 cutoff 的频率被保留，其余低频被抑制。
        mask = (radius >= self.cutoff).to(fft.dtype)
        high = torch.fft.irfft2(fft * mask, s=(height, width), norm="ortho")
        return high.abs()


class HaarWaveletHighPass(nn.Module):
    """Haar 小波高频响应。

    使用固定 Haar 滤波器提取 LH、HL、HH 三个方向的局部高频成分，
    分别对应水平、垂直、对角方向的边缘和纹理突变。
    """

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0
        weight = torch.stack([lh, hl, hh, ll], dim=0).unsqueeze(1)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, gray):
        # stride=2 实现一次下采样小波分解；这里只保留前三个高频子带。
        out = F.conv2d(gray, self.weight, stride=2)
        return out[:, :3].abs()


class ConvNeXtBlock(nn.Module):
    """轻量 ConvNeXt 风格残差块，用于频域图的局部上下文建模。"""

    def __init__(self, dim):
        super().__init__()
        # depthwise 卷积负责空间混合，pointwise 卷积负责通道混合。
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.pointwise(self.norm(self.depthwise(x)))


class MultiSourceFrequencyEncoder(nn.Module):
    """多源频域编码器。

    输入原始 RGB 图像，内部先转灰度并提取四类频域特征：
    SRM(3通道) + DCT(1通道) + FFT(1通道) + Wavelet(3通道) = 8通道。
    随后用轻量 CNN 编码并池化到 16×16，与 METER 的 image patch token 网格对齐。

    输出：
    1. freq_tokens: [B, 256, token_dim]，用于和 RGB image tokens 做 cross-attention 融合；
    2. freq_scores: [B, 256]，每个 patch 的频域异常概率，用于 Loss_freq_patch；
    3. freq_consist_feats: [B, 256, consist_dim]，用于计算频域一致性矩阵 Loss_freq_matrix。
    """

    def __init__(self, token_dim=768, hidden_dim=128, token_grid=16, consist_dim=64):
        super().__init__()
        self.token_grid = token_grid
        self.srm = FixedSRMConv()
        self.dct = DCTHighFrequencyMap()
        self.fft = FFTAmplitudeHighPass()
        self.wavelet = HaarWaveletHighPass()

        # 8通道频域图进入 CNN stem，得到 hidden_dim 通道的局部频域特征。
        self.stem = nn.Sequential(
            nn.Conv2d(8, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            ConvNeXtBlock(hidden_dim),
            ConvNeXtBlock(hidden_dim),
        )
        # 进一步编码频域上下文，保持空间分辨率，后续统一池化到 token_grid。
        self.down = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            ConvNeXtBlock(hidden_dim),
        )
        # 将 hidden_dim 维 CNN 特征投影到 METER 图像 token 的维度 token_dim。
        self.token_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, token_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(token_dim, token_dim, kernel_size=1),
        )
        # patch 级异常分数头，用 sigmoid 后得到每个 patch 的伪造概率。
        self.score_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        # 频域一致性特征头，用于构造 patch-patch 频域相似矩阵。
        self.consist_proj = nn.Conv2d(hidden_dim, consist_dim, kernel_size=1)

    def _to_gray(self, image):
        # 使用标准亮度系数将 RGB 图转换为灰度图，降低颜色冗余。
        weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (image * weights).sum(dim=1, keepdim=True)

    def _extract_frequency_maps(self, image):
        # 频域算子对数值精度较敏感，这里关闭混合精度，统一用 float 计算。
        with amp.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            gray = self._to_gray(image)
            srm = self.srm(gray).abs()
            dct = self.dct(gray)
            fft = self.fft(gray)
            wavelet = self.wavelet(gray)

            target_size = image.shape[-2:]
            maps = [
                srm,
                # DCT/FFT/Wavelet 的空间尺寸可能和原图不同，统一插值到原图大小后拼接。
                F.interpolate(dct, size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(fft, size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(wavelet, size=target_size, mode="bilinear", align_corners=False),
            ]
            freq_maps = torch.cat(maps, dim=1)
            # 逐样本标准化，避免某一类频域特征数值过大主导训练。
            mean = freq_maps.mean(dim=(-2, -1), keepdim=True)
            std = freq_maps.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            return (freq_maps - mean) / std

    def forward(self, image):
        freq_maps = self._extract_frequency_maps(image)
        feat = self.stem(freq_maps)
        feat = self.down(feat)
        # 自适应池化到 16×16，使频域 token 与 METER 的 256 个图像 patch 一一对应。
        feat_16 = F.adaptive_avg_pool2d(feat, (self.token_grid, self.token_grid))

        freq_tokens = self.token_proj(feat_16).flatten(2).transpose(1, 2)
        freq_scores = torch.sigmoid(self.score_head(feat_16).flatten(2).squeeze(1))
        freq_consist_feats = self.consist_proj(feat_16).flatten(2).transpose(1, 2)
        return freq_tokens, freq_scores, freq_consist_feats


class FrequencyGuidedFusion(nn.Module):
    """频域引导融合模块。

    该模块插在 METER image patch tokens 和 Image Intra-modal Consistency 之间。
    它以 RGB image tokens 作为 Query，以 frequency tokens 作为 Key/Value 做 cross-attention，
    再通过 gate 和可学习残差系数 alpha 控制频域信息注入强度。

    融合公式：
        fused_tokens = image_tokens + tanh(alpha) * gate * attn_out

    alpha 初始化为 0，因此训练初期该模块近似恒等映射，避免频域分支一开始破坏原模型特征。
    """

    def __init__(self, token_dim=768, hidden_dim=128, num_heads=8, token_grid=16):
        super().__init__()
        self.encoder = MultiSourceFrequencyEncoder(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            token_grid=token_grid,
        )
        self.rgb_norm = nn.LayerNorm(token_dim)
        self.freq_norm = nn.LayerNorm(token_dim)
        # RGB token 查询频域 token，从频域表示中取回与当前 patch 相关的高频异常上下文。
        self.cross_attn = nn.MultiheadAttention(token_dim, num_heads, dropout=0.0, batch_first=True)
        # 门控网络按 patch 自适应控制注入多少频域信息，降低频域噪声误导风险。
        self.gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
            nn.Sigmoid(),
        )
        # 全局可学习残差尺度；tanh 后范围稳定在 [-1, 1]。
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, image, image_tokens):
        freq_tokens, freq_scores, freq_consist_feats = self.encoder(image)
        freq_tokens = freq_tokens.to(dtype=image_tokens.dtype)
        # Query 来自 RGB token，Key/Value 来自频域 token，实现频域信息对 RGB token 的补充。
        attn_out = self.cross_attn(
            query=self.rgb_norm(image_tokens),
            key=self.freq_norm(freq_tokens),
            value=self.freq_norm(freq_tokens),
        )[0]
        # gate 的输入拼接 RGB 与频域 token，输出 [B, 256, 1] 的 patch 级融合权重。
        gate = self.gate(torch.cat([image_tokens, freq_tokens], dim=-1))
        scale = torch.tanh(self.alpha)
        fused_tokens = image_tokens + scale * gate * attn_out
        return fused_tokens, freq_scores, freq_consist_feats

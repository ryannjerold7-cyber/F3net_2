from xception import Xception
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types
from einops import rearrange
import os


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion module for fusing spatial and frequency features.
    Compatible with DCT output [B, C, H, W].
    """
    def __init__(self, dim_f, dim_g, num_heads=4):
        super().__init__()
        self.proj_f = nn.Linear(dim_f, dim_f)
        self.proj_g = nn.Linear(dim_g, dim_f)  # project G to same dim as F
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim_f, num_heads=num_heads, batch_first=True)

    def forward(self, F_spatial, G_freq):
        """
        Args:
            F_spatial: [B, C, H, W]  - spatial feature map
            G_freq:    [B, C, H, W]  - frequency feature map
        Returns:
            F_fused: [B, C, H, W]  - fused feature map
        """
        B, C, H, W = F_spatial.shape

        # Tokenize spatial and frequency features
        F_tokens = rearrange(F_spatial, 'b c h w -> b (h w) c')
        F_tokens = self.proj_f(F_tokens)

        G_tokens = rearrange(G_freq, 'b c h w -> b (h w) c')
        G_tokens = self.proj_g(G_tokens)

        # Cross-attention
        Att_out, _ = self.cross_attn(query=G_tokens, key=F_tokens, value=F_tokens)

        # Reshape back to spatial dimensions
        F_weight = rearrange(Att_out, 'b (h w) c -> b c h w', h=H, w=W)

        # Fuse with original spatial feature
        F_fused = F_spatial + F_weight

        return F_fused


class AdaptiveCrossGatingFusion(nn.Module):
    """
    Adaptive Cross-Gating Fusion (ACGF)
    —— 改进版的 MultiScaleFrequencyResidualGating
    实现双向频谱–空间门控交互：Freq ↔ Spatial
    """

    def __init__(self, num_scales=3, channels=[64, 128, 256], beta_init=1.0):
        super().__init__()
        self.num_scales = num_scales
        self.betas = nn.ParameterList([nn.Parameter(torch.tensor(beta_init)) for _ in range(num_scales)])

        # 高频残差提取卷积（频谱门控）
        self.freq_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 1, 3, padding=1),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])

        # 空间门控生成（反向作用于频谱）
        self.spatial_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[i], 1, 3, padding=1),
                nn.Sigmoid()
            ) for i in range(num_scales)
        ])

    def extract_highfreq_residual(self, freq_map):
        """执行高通滤波以提取频谱残差信息"""
        B, C, H, W = freq_map.shape
        freq = torch.fft.fft2(freq_map, norm='ortho')
        freq_shift = torch.fft.fftshift(freq)
        u = torch.arange(H, device=freq_map.device) - H // 2
        v = torch.arange(W, device=freq_map.device) - W // 2
        U, V = torch.meshgrid(u, v, indexing='ij')
        radius = torch.sqrt(U ** 2 + V ** 2)
        mask = (radius > min(H, W) * 0.25).float()
        mask = mask.unsqueeze(0).unsqueeze(0)
        freq_hp = freq_shift * mask
        img_hp = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(freq_hp)))
        return img_hp

    def forward(self, spatial_feats, freq_feats):
        out_feats = []
        for i in range(self.num_scales):
            F_spatial = spatial_feats[i]
            F_freq = freq_feats[i]

            # 对频谱做单通道均值以生成门控
            if F_freq.size(1) != 1:
                F_freq = F_freq.mean(dim=1, keepdim=True)

            # Step1: 高频残差提取
            freq_residual = self.extract_highfreq_residual(F_freq)

            # Step2: 生成频谱门控图 (用于调制空间特征)
            gate_freq = self.freq_convs[i](freq_residual)

            # Step3: 生成空间门控图 (用于调制频谱特征)
            gate_spatial = self.spatial_convs[i](F_spatial)

            # Step4: 互相调制
            F_spatial_mod = F_spatial * (1 + self.betas[i] * gate_freq)
            F_freq_mod = F_freq * (1 + self.betas[i] * gate_spatial)

            # Step5: 融合空间与频谱的双向结果
            F_mod = F_spatial_mod + 0.3 * F_freq_mod  # 控制频谱反向权重

            out_feats.append(F_mod)

        return out_feats


class F3Net(nn.Module):
    """
    F3Net with multi-scale feature fusion using CrossAttentionFusion
    Supports loading pretrained Xception weights for the backbone
    """

    def __init__(self, num_classes=1, img_width=299, img_height=299, mode='Mix',device=None,pretrained_path=r'../xception.pth'):
        super(F3Net, self).__init__()
        self.num_classes = num_classes

        # --- Backbone: Xception that returns three-scale features ---
        self.backbone = Xception(num_classes)

        # --- Load pretrained weights if available ---
        if pretrained_path is not None and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path)

            # reshape pointwise convs if needed
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            # remove fc weights
            state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}

            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded pretrained weights, missing keys: {missing}, unexpected: {unexpected}")

        # --- Patch DCT module ---
        self.patch_dct = multi_DCT

        # --- CrossAttentionFusion modules for each scale ---
        self.fuse_high = CrossAttentionFusion(dim_f=728, dim_g=728, num_heads=4)
        self.fuse_mid = CrossAttentionFusion(dim_f=728, dim_g=728, num_heads=4)
        self.fuse_low = CrossAttentionFusion(dim_f=1024, dim_g=1024, num_heads=4)

        # --- MFRG (Frequency Residual Gating) ---
        self.mfrg = AdaptiveCrossGatingFusion(num_scales=3, channels=[728, 728, 1024], beta_init=1.0)

        # --- Classification head ---
        total_channels = 728 + 728 + 1024
        self.fc = nn.Linear(total_channels, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Extract multi-scale features from backbone
        F_high, F_mid, F_low = self.backbone.features(x)

        # Apply DCT to each scale feature map
        G_high = self.patch_dct(F_high, patch_size=8, stride=4)
        G_mid = self.patch_dct(F_mid, patch_size=8, stride=4)
        G_low = self.patch_dct(F_low, patch_size=8, stride=4)

        # Frequency residual gating per scale
        spatial_list = [F_high, F_mid, F_low]
        freq_for_gating = [G_high.mean(dim=1, keepdim=True),
                           G_mid.mean(dim=1, keepdim=True),
                           G_low.mean(dim=1, keepdim=True)]
        F_high_mod, F_mid_mod, F_low_mod = self.mfrg(spatial_list, freq_for_gating)

        # Cross-attention fusion for each scale
        F_high_fused = self.fuse_high(F_high_mod, G_high)
        F_mid_fused = self.fuse_mid(F_mid_mod, G_mid)
        F_low_fused = self.fuse_low(F_low_mod, G_low)

        # Global average pooling
        F_high_pooled = F.adaptive_avg_pool2d(F_high_fused, (1, 1))
        F_mid_pooled = F.adaptive_avg_pool2d(F_mid_fused, (1, 1))
        F_low_pooled = F.adaptive_avg_pool2d(F_low_fused, (1, 1))

        # Flatten and concatenate
        F_high_flat = F_high_pooled.view(F_high_pooled.size(0), -1)
        F_mid_flat = F_mid_pooled.view(F_mid_pooled.size(0), -1)
        F_low_flat = F_low_pooled.view(F_low_pooled.size(0), -1)
        multi_scale_features = torch.cat([F_high_flat, F_mid_flat, F_low_flat], dim=1)

        # Classification
        features = self.dropout(multi_scale_features)
        output = self.fc(features)

        # Return both prediction & features for FCL
        return {
            'multi_scale_features': multi_scale_features,
            'output': output,
            'spatial_feats': [F_high_fused, F_mid_fused, F_low_fused],
            'freq_feats': [G_high, G_mid, G_low]
        }


# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

def multi_DCT(feature_map, patch_size=8, stride=4):
    """
    Apply sliding window DCT to feature maps

    Args:
        feature_map: Input feature map [B, C, H, W]
        patch_size: Size of DCT patch (default: 8)
        stride: Stride for sliding window (default: 4)

    Returns:
        DCT spectrum map with same spatial dimensions as input
    """
    import torch.nn.functional as F

    B, C, H, W = feature_map.shape

    # Calculate padding to ensure we can extract patches
    pad_h = (patch_size - (H % patch_size)) % patch_size
    pad_w = (patch_size - (W % patch_size)) % patch_size

    # Pad the feature map
    if pad_h > 0 or pad_w > 0:
        feature_map = F.pad(feature_map, (0, pad_w, 0, pad_h), mode='reflect')
        H_padded, W_padded = H + pad_h, W + pad_w
    else:
        H_padded, W_padded = H, W

    # Create DCT matrix for the patch size
    dct_matrix = torch.tensor(DCT_mat(patch_size), dtype=torch.float32, device=feature_map.device)
    dct_matrix_T = dct_matrix.transpose(0, 1)

    # Use unfold to extract patches
    unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=stride, padding=0)
    patches = unfold(feature_map)  # [B, C*patch_size*patch_size, num_patches]

    # Reshape patches for DCT computation
    num_patches = patches.shape[2]
    patches = patches.view(B, C, patch_size, patch_size, num_patches)
    patches = patches.permute(0, 4, 1, 2, 3)  # [B, num_patches, C, patch_size, patch_size]

    # Apply DCT to each patch
    dct_patches = torch.zeros_like(patches)
    for i in range(num_patches):
        patch = patches[:, i, :, :, :]  # [B, C, patch_size, patch_size]
        # Apply DCT: DCT_matrix @ patch @ DCT_matrix_T
        dct_patch = torch.matmul(torch.matmul(dct_matrix, patch), dct_matrix_T)
        dct_patches[:, i, :, :, :] = dct_patch

    # Reshape back to spatial format
    dct_patches = dct_patches.permute(0, 2, 3, 4, 1)  # [B, C, patch_size, patch_size, num_patches]

    # Calculate output dimensions
    out_h = (H_padded - patch_size) // stride + 1
    out_w = (W_padded - patch_size) // stride + 1

    # Reconstruct the DCT spectrum map
    dct_spectrum = torch.zeros(B, C, H_padded, W_padded, device=feature_map.device)
    count_map = torch.zeros(H_padded, W_padded, device=feature_map.device)

    for i in range(out_h):
        for j in range(out_w):
            start_h = i * stride
            start_w = j * stride
            end_h = start_h + patch_size
            end_w = start_w + patch_size

            patch_idx = i * out_w + j
            dct_spectrum[:, :, start_h:end_h, start_w:end_w] += dct_patches[:, :, :, :, patch_idx]
            count_map[start_h:end_h, start_w:end_w] += 1

    # Average overlapping regions
    count_map = count_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    dct_spectrum = dct_spectrum / (count_map + 1e-8)

    # Crop back to original size
    dct_spectrum = dct_spectrum[:, :, :H, :W]

    return dct_spectrum


def get_xcep_state_dict(pretrained_path='../xception.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict


class MixBlock(nn.Module):
    # An implementation of the cross attention module in F3-Net
    def __init__(self, c_in, width, height):
        super(MixBlock, self).__init__()
        self.FAD_query = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_query = nn.Conv2d(c_in, c_in, (1,1))

        self.FAD_key = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_key = nn.Conv2d(c_in, c_in, (1,1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))

        self.FAD_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.FAD_bn = nn.BatchNorm2d(c_in)
        self.LFS_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.LFS_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_FAD, x_LFS):
        B, C, W, H = x_FAD.size()
        assert W == H

        q_FAD = self.FAD_query(x_FAD).view(-1, W, H)    # [BC, W, H]
        q_LFS = self.LFS_query(x_LFS).view(-1, W, H)
        M_query = torch.cat([q_FAD, q_LFS], dim=2)  # [BC, W, 2H]

        k_FAD = self.FAD_key(x_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_LFS = self.LFS_key(x_LFS).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_FAD, k_LFS], dim=1)    # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  #[BC, W, W]
        attention = self.softmax(energy).view(B, C, W, W)

        att_LFS = x_LFS * attention * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_LFS))

        att_FAD = x_FAD * attention * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_FAD))
        return y_FAD, y_LFS

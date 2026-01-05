import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
import torchvision
from torch.nn import GroupNorm


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """

    def __init__(self, shape, base=500):
        super(RoPE, self).__init__()
        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0
        # angles
        theta_ks = torch.arange(k_max).float()
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)
        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super(LinearAttention, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h = self.input_resolution[0]
        w = self.input_resolution[1]
        num_heads = self.num_heads
        head_dim = c // num_heads
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


class MILABlock(nn.Module):
    r""" MLLA Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super(MILABlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)
        # Linear Attention
        x = self.attn(x)
        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"


class LightConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(LightConvBlock, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # 深度卷积
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # 逐点卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            LightConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = LightConvBlock(in_channels, out_channels)
        self.up_adapts = nn.ModuleList([
            AdaptiveChannelModulation(256),
            AdaptiveChannelModulation(128),
            AdaptiveChannelModulation(64)
        ])

    def forward(self, x1, x2, id=None, use_simam=False):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)

        if id is not None:
            x = self.up_adapts[id](x, use_simam)

        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AdaptiveChannelModulation(nn.Module):
    def __init__(self, dim):
        super(AdaptiveChannelModulation, self).__init__()
        self.gamma_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        self.beta_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Tanh()
        )
        self.simam = SimAM()

    def forward(self, x, use_simam=False):
        if use_simam:
            x = self.simam(x)
        gamma = self.gamma_net(x)
        beta = self.beta_net(x)
        return x * gamma + beta


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)


class HierarchicalFeatureEnhancementModule(nn.Module):
    def __init__(self, in_channels_list=[256, 128, 64], out_channels=64, target_size=(609, 387)):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.deform_align = nn.ModuleList()
        self.target_size = target_size

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.GroupNorm(8, out_channels),
                    nn.ReLU()
                )
            )
            self.deform_align.append(
                DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

        self.weights = nn.Parameter(torch.ones(len(in_channels_list)))
        self.spatial_att = SpatialAttention()

        self.frb = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels, 3, padding=1),
            GroupNorm(8, out_channels),
            SpatialAttention(),
            ChannelAttention(out_channels)
        )

    def forward(self, features):
        aligned_feats = []
        for i, feat in enumerate(features):
            x = self.lateral_convs[i](feat)
            if x.size()[-2:] != self.target_size:
                x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)

            x = self.deform_align[i](x)

            aligned_feats.append(x)

        weights = F.relu(self.weights)
        weights = weights / weights.sum() + 1e-6

        fused_feat = torch.zeros_like(aligned_feats[0])
        for i, feat in enumerate(aligned_feats):
            att_feat = self.spatial_att(feat) * feat
            fused_feat = fused_feat + weights[i] * att_feat

        # fused_feat = self.frb(fused_feat)

        return fused_feat


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, padding=padding)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x):
        offset = self.offset_conv(x)
        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.conv.weight,
            bias=self.conv.bias,
            stride=1,
            padding=self.padding,
            dilation=1
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        att = torch.sigmoid(self.conv(concat))
        return x * att


class DP_HAFNet(nn.Module):
    def __init__(self, n_channels, n_classes_list, bilinear=True, img_size=(609, 387)):
        super(DP_HAFNet, self).__init__()
        self.n_classes_list = n_classes_list
        self.n_channels = n_channels
        self.bilinear = bilinear

        resolutions = [(img_size[0] // 16, img_size[1] // 16),  # down4输出尺寸
                       (img_size[0] // 8, img_size[1] // 8),  # down3
                       (img_size[0] // 4, img_size[1] // 4),
                       (img_size[0] // 2, img_size[1] // 2)]  # down2

        self.bottleneck_mlla = MILABlock(512, resolutions[0], 4)
        self.skip_mllas = nn.ModuleList([
            MILABlock(512, resolutions[1], 4),
            MILABlock(256, resolutions[2], 4),
            MILABlock(128, resolutions[3], 2)
        ])

        self.down_adapts = nn.ModuleList([
            AdaptiveChannelModulation(64),
            AdaptiveChannelModulation(128),
            AdaptiveChannelModulation(256),
            AdaptiveChannelModulation(512)
        ])

        self.bottleneck_adapt = AdaptiveChannelModulation(512)  # 瓶颈层
        self.output_adapt = AdaptiveChannelModulation(64)

        self.inc = LightConvBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc1 = OutConv(64, n_classes_list[0])
        self.outc2 = OutConv(64, n_classes_list[1])

        self.task1_fusion = HierarchicalFeatureEnhancementModule(
            in_channels_list=[256, 128, 64],
            out_channels=64
        )
        self.frb = nn.Sequential(
            DepthwiseSeparableConv(128, 64, 3, padding=1),
            GroupNorm(8, 64),  # 标准化特征分布
            SpatialAttention(),  # 先聚焦空间关键区域
            ChannelAttention(64)
        )

    def _apply_mlla(self, x, mlla_block):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,H*W,C]
        x = mlla_block(x)  # MLLA处理
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # 恢复形状
        return x

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.down_adapts[0](x1, True)

        x2 = self.down1(x1)
        x2 = self.down_adapts[1](x2, True)

        x3 = self.down2(x2)
        x3 = self.down_adapts[2](x3, True)

        x4 = self.down3(x3)
        x4 = self.down_adapts[3](x4, True)

        x5 = self.down4(x4)
        x5 = self.bottleneck_adapt(x5, True)
        x5 = self._apply_mlla(x5, self.bottleneck_mlla)

        decoder_feats = []

        x_branch1 = self.up1(x5, self._apply_mlla(x4, self.skip_mllas[0]), 0, use_simam=True)
        decoder_feats.append(x_branch1)
        x_branch1 = self.up2(x_branch1, self._apply_mlla(x3, self.skip_mllas[1]), 1, use_simam=True)
        decoder_feats.append(x_branch1)
        x_branch1 = self.up3(x_branch1, self._apply_mlla(x2, self.skip_mllas[2]), 2, use_simam=True)
        decoder_feats.append(x_branch1)
        x_branch1 = self.up4(x_branch1, x1)
        x_branch1 = self.output_adapt(x_branch1)

        task1_feat = self.task1_fusion(decoder_feats)
        x1 = torch.cat([task1_feat, x_branch1], dim=1)
        x1 = self.frb(x1)

        return self.outc1(x1), self.outc2(x_branch1)

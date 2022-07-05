import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.jit
import math
from einops import rearrange


class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., cha_sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.cha_sr_ratio = cha_sr_ratio if num_heads > 1 else 1

        self.qv = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def _gen_attn(self, q, k):
        q = q.softmax(-2).transpose(-1,-2)
        _, _, N, _  = k.shape
        k = torch.nn.functional.adaptive_avg_pool2d(k.softmax(-2), (N, 1))
        
        attn = torch.nn.functional.sigmoid(q @ k)
        return attn  * self.temperature
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b (h w) c')
        B, N, C = x.shape
                
        qv = self.qv(x).reshape(B, N, C, 2).permute(3, 0, 1, 2)
        q = qv[0].reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = qv[1].reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)

        repeat_time = N // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(N)))
        
        return x

    
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}
    
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gamma = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features, in_features, kernel_size=kernel_size, padding=padding, groups=in_features)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(
            in_features, out_features, kernel_size=kernel_size, padding=padding, groups=out_features)

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)


        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x

# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(   nn.Linear(2, 512, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
            
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

# --------------------------------------------
# 1x1 convolution layer
# --------------------------------------------
def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """ 1x1 convolution

    Input shape:
        (batch size, in_planes, height, width)
    Output shape:
        (batch size, out_planes, height, width)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False)



class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """
    
    def __init__(   self, 
                    dim, 
                    dim_out,
                    heads, 
                    input_resolution,
                    window_size=7, 
                    shift_size=0,
                    qkv_bias=True, 
                    drop=0., 
                    attn_drop=0., 
                    drop_path=0.,
                    norm_layer=nn.BatchNorm2d, 
                    pretrained_window_size=0,
                    activation=nn.GELU):
        
        super().__init__()
        self.dim = dim
        self.num_heads = heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.expansion = 1
        

        self.shortcut = []
        if dim != dim_out * self.expansion:
            self.shortcut.append(conv1x1(dim, dim_out * self.expansion, stride=1))
            self.norm1 = norm_layer(dim)
            
        self.shortcut = nn.Sequential(*self.shortcut)
        
        # if min(input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        
        self.activation = activation()
        
        self.attn = WindowAttention(
            dim, 
            window_size=to_2tuple(self.window_size), 
            num_heads=heads,
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = input_resolution
            
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        self.chnnel_processing = ChannelProcessing(
                                                    dim, num_heads=heads, qkv_bias=qkv_bias, 
                                                    attn_drop=attn_drop
                                                )
        self.norm3 = norm_layer(dim)

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, C, H, W = x.size()

        if len(self.shortcut) > 0:
            x = self.norm1(x)
            x = self.activation(x)
            shortcut = self.shortcut(x)

        else:
            shortcut = x

        x = x.permute(0, 2, 3, 1)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.permute(0 ,3, 1, 2)
        
        x = shortcut + self.drop_path(self.norm2(x))

        shortcut2 = x
        
        x = shortcut2 + self.drop_path(self.norm3(self.chnnel_processing(x)))
        
        return x



"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021). 
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.apply(self._init_weights)
                    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(   self, 
                    cfgs,
                    block = MBConv,
                    block2 = SwinTransformerBlock,
                    input_resolution=(112, 112),
                    n_classes=1000, 
                    width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 1)]
        # building inverted residual blocks
        block = MBConv
        
        for t, c, n, s, use_se, n_transformer, heads in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers = layers + self.stack_layers(
                                                block, 
                                                block2, 
                                                input_channel=input_channel, 
                                                output_channel=output_channel, 
                                                blocks=n,
                                                blocks2=n_transformer,
                                                heads=heads,
                                                input_resolution=(input_resolution[0], input_resolution[1]), 
                                                window_size=7, 
                                                stride=s,
                                                expand_ratio=t,
                                                use_se=use_se)
            
            input_channel = output_channel
            input_resolution = (input_resolution[0]//2, input_resolution[1]//2)
                
        self.features = nn.Sequential(*layers)
        
        # self.dropout = nn.Dropout()
        self.gap = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(output_channel * 7 * 7, n_classes)
        self.bn = nn.BatchNorm1d(n_classes)
        
        self.apply(self._init_weights)
                    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.features(x)
        
        # x = self.dropout(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        
        return x

                
    def stack_layers(   self, 
                        block, 
                        block2, 
                        input_channel, 
                        output_channel, 
                        blocks, 
                        blocks2, 
                        heads, 
                        input_resolution, 
                        window_size=7, 
                        stride=1,
                        expand_ratio=1,
                        use_se=True
                        ):
        # Peforms downsample if stride != 1 of inplanes != block.expansion
        
        if input_resolution[0] > window_size:
            num_blocks = (2*(blocks//3) + (blocks%3) - 1)
            
            assert 2*blocks2+blocks2 <= blocks, 'The number of transformers must not exceed cnn !!!'
            
        else:
            num_blocks = blocks - 1
            
            assert 2*blocks2 <= blocks, 'The number of transformers must not exceed cnn !!!'
        
        
        alt_seq = [False] * num_blocks
        for i in range(blocks2):
            idx = -2*(i)-1
            alt_seq[idx] = True
        
        # print(alt_seq)
        
        layers = []
        # For the first residual block, stride is 1 or 2
        layers.append(block(input_channel, output_channel, stride, expand_ratio, use_se))
        
        # From the second residual block, stride is 1
        for is_alt in alt_seq:
            if not is_alt:
                layers.append(block(output_channel, output_channel, 1, expand_ratio, use_se))
            else:
                layers.append(block2(output_channel, output_channel, heads=heads, input_resolution=input_resolution))
                if input_resolution[0] > window_size:
                    layers.append(block2(output_channel, output_channel, heads=heads, input_resolution=input_resolution, shift_size=window_size//2))

        return layers




def EffiAlter_s(conf):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE    # n_transformer, heads
        [1,  24,  1, 1, 0] +        [0, 2],
        [2,  48,  3, 2, 0] +        [0, 2],
        [2,  64,  4, 2, 0] +        [0, 2],
        [2, 128,  5, 2, 1] +        [0, 4],
        [3, 160,  9, 1, 1] +        [0, 8],
        [3, 256, 4, 2, 1] +        [0, 8],
        # # t, c, n, s, SE    # n_transformer, heads
        # [1,  24,  2, 1, 0] +        [0, 2],
        # [4,  48,  4, 2, 0] +        [0, 2],
        # [4,  64,  4, 2, 0] +        [0, 2],
        # [4, 128,  6, 2, 1] +        [0, 4],
        # [6, 160,  9, 1, 1] +        [0, 8],
        # [6, 256, 15, 2, 1] +        [0, 8],
    ]
    return EffNetV2(cfgs, n_classes=conf.emd_size)


def EffiAlter_m(conf):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE    # n_transformer, heads
        [1,  24,  3, 1, 0] +        [0, 2],
        [4,  48,  5, 2, 0] +        [0, 2],
        [4,  80,  5, 2, 0] +        [0, 2],
        [4, 160,  7, 2, 1] +        [0, 4],
        [6, 176, 14, 1, 1] +        [0, 8],
        [6, 304, 18, 2, 1] +        [2, 8],
        [6, 512,  5, 1, 1] +        [2, 16],
    ]
    return EffNetV2(cfgs, n_classes=conf.emd_size)


def EffiAlter_l(conf):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE    # n_transformer, heads
        [1,  32,  4, 1, 0] +        [0, 2],
        [4,  64,  7, 2, 0] +        [0, 2],
        [4,  96,  7, 2, 0] +        [0, 2],
        [4, 192, 10, 2, 1] +        [0, 4],
        [6, 224, 19, 1, 1] +        [2, 8],
        [6, 384, 25, 2, 1] +        [2, 16],
        [6, 640,  7, 1, 1] +        [2, 32],
    ]
    return EffNetV2(cfgs, n_classes=conf.emd_size)


def EffiAlter_xl(conf):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE    # n_transformer, heads
        [1,  32,  4, 1, 0] +        [0, 2],
        [4,  64,  8, 2, 0] +        [0, 2],
        [4,  96,  8, 2, 0] +        [0, 2],
        [4, 192, 16, 2, 1] +        [0, 4],
        [6, 256, 24, 1, 1] +        [2, 8],
        [6, 512, 32, 2, 1] +        [2, 16],
        [6, 640,  8, 1, 1] +        [2, 32],
    ]
    return EffNetV2(cfgs, n_classes=conf.emd_size)





def Encoder(conf):
    if conf.network == 'EffiAlter_s':
        return EffiAlter_s(conf)
    elif conf.network == 'EffiAlter_m':
        return EffiAlter_m(conf)
    elif conf.network == 'EffiAlter_l':
        return EffiAlter_l(conf)
    elif conf.network == 'EffiAlter_xl':
        return EffiAlter_xl(conf)
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.jit
import math

class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., linear=False, drop_path=0., 
                 mlp_hidden_dim=None, act_layer=nn.GELU, drop=None, norm_layer=nn.LayerNorm, cha_sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.cha_sr_ratio = cha_sr_ratio if num_heads > 1 else 1

        # config of mlp for v processing
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = Mlp(in_features=dim//self.cha_sr_ratio, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        self.norm_v = norm_layer(dim//self.cha_sr_ratio)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
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
    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape
        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd*Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)

        repeat_time = N // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x,  (attn * v.transpose(-1, -2)).transpose(-1, -2) #attn
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
        elif isinstance(m, nn.LayerNorm):
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

    def __init__(self, dim, window_size, num_heads, dim_head=32, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        assert dim_head * self.num_heads == dim, 'Not match dim_head * num_heads and hidden_dim'

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
        self.shortcut = nn.Sequential(*self.shortcut)
            
        self.norm1 = norm_layer(dim)
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
            H, W = self.input_resolution
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
                                                    attn_drop=attn_drop, drop_path=drop_path, drop=drop, 
                                                    mlp_hidden_dim = int(dim * 4)
                                                )

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

        return x


# --------------------------------------------
# 7x7 convolution layer
# --------------------------------------------
def conv7x7(in_planes, out_planes, stride=1, groups=1):
    """ 7x7 convolution with padding = 3

    Input shape:
        (batch size, in_planes, height, width)
    Output shape:
        (batch size, out_planes, height, width)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, groups=groups,
                    bias=False)

# --------------------------------------------
# 3x3 convolution layer
# --------------------------------------------
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 3x3 convolution with padding = 1

    Input shape:
        (batch size, in_planes, height, width)
    Output shape:
        (batch size, out_planes, height, width)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups,
                    bias=False, dilation=dilation)

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
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, groups=groups,
                 bias=False)


# ======================================= Classifier layers ========================================

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def output_layer(emb_size, channel, kernel_size, dropout_rate=0.0):

    filter_size = kernel_size * kernel_size
    output = nn.Sequential(
        nn.BatchNorm2d(channel),
        nn.Dropout(dropout_rate),
        Flatten(),
        nn.Linear(channel * filter_size, emb_size),
        nn.BatchNorm1d(emb_size),
    )

    return output

# ======================================== Residual Network ========================================

# --------------------------------------------
# Residual blocks - BasicBlock
# --------------------------------------------
class BasicBlock(nn.Module):
    """ Implement of Residual block - IR BasicBlock architecture (https://arxiv.org/pdf/1610.02915.pdf):

    This layer creates a basic residual block of AlterNet architecture, which is a pre-activation Residual Unit.
    It consists of two 3x3 convolution layers, three batch normalization layers and one ReLU layers.

    Shortcut connection options:
        If the output feature map has the same dimensions as the input feature map,
        the shortcut performs identity mapping.

        If the output feature map dimensions increase(usually doubled),
        the shortcut performs downsample to match dimensions and halve the feature map size
        by using 1x1 convolution with stride 2.

    Args:
        inplanes: dimension of input feature
        planes: dimension of ouput feature
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # When the dimensions of the output feature map increase,
        # self.conv2 and self.downsample layers performs downsample
        
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

# --------------------------------------------
# Residual blocks - Bottleneck
# --------------------------------------------
class Bottleneck(nn.Module):
    """ Implement of Residual block - IR Bottleneck architecture (https://arxiv.org/pdf/1610.02915.pdf):

    This layer creates a bottleneck residual block of AlterNet architecture, which is a original Residual Unit.
    It consists of three 3x3 convolution layers, three batch normalization layers and three ReLU layers.

    Shortcut connection options:
        If the output feature map has the same dimensions as the input feature map,
        the shortcut performs identity mapping.

        If the output feature map dimensions increase(usually doubled),
        the shortcut performs downsample to match dimensions and halve the feature map size
        by using 1x1 convolution with stride 2.

    Args:
        inplanes: dimension of input feature
        planes: compressed dimension after passing conv1
        expansion: ratio to expand dimension
    """
    
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # When the dimensions of the output feature map increase,
        # self.conv2 and self.downsample layers performs downsample
        
        self.conv1 = conv1x1(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn4 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

# --------------------------------------------
# AlterNet backbone
# --------------------------------------------
class AlterNet(nn.Module):
    """ Implement of Residual network (AlterNet) (https://arxiv.org/pdf/1512.03385.pdf):

    This layer creates a AlterNet model by stacking basic or bottleneck residual blocks.

    Args:
        block: block to stack in each layer - BasicBlock or Bottleneck
        layers: # of stacked blocks in each layer
    """
    def __init__(self, conf, 
                    block, block2,
                    num_blocks, num_blocks2,
                    heads
                    ):
        # options from configuration
        super(AlterNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = conv3x3(3, self.inplanes, stride=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.stack_layers(block, block2, 64, num_blocks[0], num_blocks2[0], heads[0])
        self.layer2 = self.stack_layers(block, block2,  128, num_blocks[1], num_blocks2[1], heads[1], stride=2)
        self.layer3 = self.stack_layers(block, block2,  256, num_blocks[2], num_blocks2[2], heads[2], stride=2)
        self.layer4 = self.stack_layers(block, block2,  conf.emd_size, num_blocks[3], num_blocks2[3], heads[3], stride=2)

        self.bn2 = nn.BatchNorm2d(block.expansion * conf.emd_size)
        self.dropout = nn.Dropout()
        self.gap = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(block.expansion * conf.emd_size * 7 * 7, conf.emd_size)
        self.bn3 = nn.BatchNorm1d(conf.emd_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def stack_layers(self, block, block2, planes, blocks, blocks2, heads, stride=1):
        downsample = None
        # Peforms downsample if stride != 1 of inplanes != block.expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        alt_seq = [False] * (blocks - blocks2 * 2 - 1) + [False, True] * blocks2
        layers = []
        # For the first residual block, stride is 1 or 2
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        # From the second residual block, stride is 1
        self.inplanes = planes * block.expansion
        for is_alt in alt_seq:
            if not is_alt:
                layers.append(block(self.inplanes, planes))
            else:
                layers.append(block2(self.inplanes, planes, heads=heads))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        
        return x

# --------------------------------------------
# Define AlterNet models
# --------------------------------------------
def AlterNet18(conf, **kwargs):
    """ Constructs a AlterNet-18 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    MSABlock = SwinTransformerBlock
    model = AlterNet(conf, ResidualBlock, MSABlock, 
                        num_blocks=[2, 2, 2, 2], num_blocks2=[0, 1, 1, 1], 
                        heads=(2, 4, 8, 16),
                        **kwargs)

    
    return model

def AlterNet34(conf, **kwargs):
    """ Constructs a AlterNet-34 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    MSABlock = SwinTransformerBlock
    model = AlterNet(conf, ResidualBlock, MSABlock, 
                        num_blocks=[3, 4, 6, 3], num_blocks2=[0, 1, 3, 2], 
                        heads=(2, 4, 8, 16),
                        **kwargs)

    
    return model

def AlterNet50(conf, **kwargs):
    """ Constructs a AlterNet-50 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    MSABlock = SwinTransformerBlock
    model = AlterNet(conf, ResidualBlock, MSABlock, 
                        num_blocks=[3, 4, 14, 3], num_blocks2=[0, 1, 7, 2], 
                        heads=(2, 4, 8, 16),
                        **kwargs)

    
    return model

def AlterNet100(conf, **kwargs):
    """ Constructs a AlterNet-100 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    MSABlock = SwinTransformerBlock
    model = AlterNet(conf, ResidualBlock, MSABlock, 
                        num_blocks=[3, 13, 30, 4], num_blocks2= [0, 1, 1, 2],
                        heads=(2, 4, 8, 16),
                        **kwargs)

    
    return model

def AlterNet200(conf, **kwargs):
    """ Constructs a AlterNet-100 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    MSABlock = SwinTransformerBlock
    model = AlterNet(conf, ResidualBlock, MSABlock, 
                        num_blocks=[3, 43, 50, 3], num_blocks2= [0, 1, 3, 2],
                        heads=(2, 4, 8, 16),
                        **kwargs)

    
    return model

def Encoder(conf):
    if conf.network == 'AlterNet200':
        return AlterNet200(conf)
    elif conf.network == 'AlterNet100':
        return AlterNet100(conf)
    elif conf.network == 'AlterNet50':
        return AlterNet50(conf)
    elif conf.network == 'AlterNet34':
        return AlterNet34(conf)
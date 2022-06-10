import torch
import torch.nn as nn

# ======================================= Convolution layers =======================================
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
import types
import math
import torch.nn.functional as F

from torch import einsum


class FeedForward(nn.Module):

    def __init__(self, dim_in, hidden_dim, dim_out=None, *,
                 dropout=0.0,
                 f=nn.Linear, activation=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.net = nn.Sequential(
            f(dim_in, hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            f(hidden_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention1d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn


class Attention2d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, k=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _, y = x.shape
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)

        out = self.to_out(out)

        return out, attn


class Transformer(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dim_mlp=1024, dropout=0.0, sd=0.0,
                 attn=Attention1d, norm=nn.LayerNorm,
                 f=nn.Linear, activation=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.shortcut = []
        if dim_in != dim_out:
            self.shortcut.append(norm(dim_in))
            self.shortcut.append(nn.Linear(dim_in, dim_out))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.norm1 = norm(dim_in)
        self.attn = attn(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sd1 = DropPath(sd) if sd > 0.0 else nn.Identity()

        self.norm2 = norm(dim_out)
        self.ff = FeedForward(dim_out, dim_mlp, dim_out, dropout=dropout, f=f, activation=activation)
        self.sd2 = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        skip = self.shortcut(x)
        x = self.norm1(x)
        x, attn = self.attn(x, mask=mask)
        x = self.sd1(x) + skip

        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.sd2(x) + skip

        return x
    

class BNGAPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(BNGAPBlock, self).__init__()

        self.bn = bn(in_features)
        self.relu = relu()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = dense(in_features, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)

        return x


class MLPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(MLPBlock, self).__init__()

        self.dense1 = dense(in_features, 4096)
        self.relu1 = relu()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense2 = dense(4096, 4096)
        self.relu2 = relu()
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense3 = dense(4096, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return x
    

class LocalAttention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 window_size=7, k=1,
                 heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        self.attn = Attention2d(dim_in, dim_out,
                                heads=heads, dim_head=dim_head, dropout=dropout, k=k)
        self.window_size = window_size

        self.rel_index = self.rel_distance(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        if h == 4:
            self.window_size = 2
            self.rel_index = self.rel_distance(self.window_size) + self.window_size - 1
            
        p = self.window_size
        n1 = h // p
        n2 = w // p

        mask = torch.zeros(p ** 2, p ** 2, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]

        x = rearrange(x, "b c (n1 p1) (n2 p2) -> (b n1 n2) c p1 p2", p1=p, p2=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, "(b n1 n2) c p1 p2 -> b c (n1 p1) (n2 p2)", n1=n1, n2=n2, p1=p, p2=p)

        return x, attn

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        d = i[None, :, :] - i[:, None, :]

        return d
    
    
class AttentionBlockB(nn.Module):
    # Attention block with pre-activation.
    # We use this block by default.
    expansion = 4

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, sd=0.0,
                 stride=1, window_size=7, k=1, norm=nn.BatchNorm2d, activation=nn.GELU,
                 **block_kwargs):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        attn = partial(LocalAttention, window_size=window_size, k=k)
        width = dim_in // self.expansion

        self.shortcut = []
        if stride != 1 or dim_in != dim_out * self.expansion:
            self.shortcut.append(conv1x1(dim_in, dim_out * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.norm1 = norm(dim_in)
        self.relu = activation()

        self.conv = nn.Conv2d(dim_in, width, kernel_size=1, bias=False)
        self.norm2 = norm(width)
        self.attn = attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.norm1(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.conv(x)
        x = self.norm2(x)
        x, attn = self.attn(x)

        x = self.sd(x) + skip

        return x


class AttentionBasicBlockB(AttentionBlockB):
    expansion = 1

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
                    heads, cblock=BNGAPBlock,
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
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
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
    MSABlock = AttentionBasicBlockB
    model = AlterNet(conf, ResidualBlock, MSABlock, 
                        num_blocks=[2, 2, 2, 2], num_blocks2=[0, 1, 1, 2], 
                        heads=(3, 6, 12, 24),
                        **kwargs)

    
    return model

def AlterNet34(conf, **kwargs):
    """ Constructs a AlterNet-34 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    MSABlock = AttentionBasicBlockB
    model = AlterNet(conf, ResidualBlock, MSABlock, 
                        num_blocks=[3, 4, 6, 3], num_blocks2=[0, 1, 1, 2], 
                        heads=(3, 6, 12, 24),
                        **kwargs)

    
    return model

def AlterNet50(conf, **kwargs):
    """ Constructs a AlterNet-50 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    MSABlock = AttentionBasicBlockB
    model = AlterNet(conf, ResidualBlock, MSABlock, 
                        num_blocks=[3, 4, 14, 3], num_blocks2=[0, 1, 1, 2], 
                        heads=(3, 6, 12, 24),
                        **kwargs)

    
    return model

def AlterNet100(conf, **kwargs):
    """ Constructs a AlterNet-100 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    MSABlock = AttentionBasicBlockB
    model = AlterNet(conf, ResidualBlock, MSABlock, 
                        num_blocks=[3, 13, 30, 3], num_blocks2= [0, 1, 1, 2],
                        heads=(3, 6, 12, 24),
                        **kwargs)

    
    return model

def AlterNet200(conf, **kwargs):
    """ Constructs a AlterNet-100 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    MSABlock = AttentionBasicBlockB
    model = AlterNet(conf, ResidualBlock, MSABlock, 
                        num_blocks=[3, 43, 50, 3], num_blocks2= [0, 1, 1, 2],
                        heads=(3, 6, 12, 24),
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
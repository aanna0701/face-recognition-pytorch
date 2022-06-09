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



class GAPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GAPBlock, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = dense(in_features, num_classes)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)

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


class GMaxPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GMaxPBlock, self).__init__()

        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.dense = dense(in_features, num_classes)

    def forward(self, x):
        x = self.gmp(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)

        return x


class GMedPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GMedPBlock, self).__init__()

        self.dense = dense(in_features, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], x.size()[1], -1)
        x = torch.topk(x, k=int(x.size()[2] / 2), dim=2)[0][:, :, -1]
        x = x.view(x.size()[0], -1)
        x = self.dense(x)

        return x


class GAPClipBlock(nn.Module):

    def __init__(self, in_features, num_classes, temp=2e0, **kwargs):
        super(GAPClipBlock, self).__init__()

        self.temp = temp
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = dense(in_features, num_classes)

    def forward(self, x):
        x = self.gap(x)
        x = self.temp * (F.sigmoid(x / self.temp) - 0.5)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)

        return x


class GAPMLPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GAPMLPBlock, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = dense(in_features, 4096)
        self.relu1 = relu()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense2 = dense(4096, 4096)
        self.relu2 = relu()
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense3 = dense(4096, num_classes)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels,
                 stride=1, groups=1, width_per_group=64, sd=0.0,
                 **block_kwargs):
        super(BasicBlock, self).__init__()

        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        width = int(channels * (width_per_group / 64.)) * groups

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(conv1x1(in_channels, channels * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.bn = bn(in_channels)
        self.relu = relu()

        self.conv1 = conv3x3(in_channels, width, stride=stride)
        self.conv2 = nn.Sequential(
            bn(width),
            relu(),
            conv3x3(width, channels * self.expansion),
        )

        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.bn(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.bn(x)
            x = self.relu(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.sd(x) + skip

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels,
                 stride=1, groups=1, width_per_group=64, sd=0.0,
                 **block_kwargs):
        super(Bottleneck, self).__init__()

        width = int(channels * (width_per_group / 64.)) * groups

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(conv1x1(in_channels, channels * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.bn = bn(in_channels)
        self.relu = relu()

        self.conv1 = conv1x1(in_channels, width)
        self.conv2 = nn.Sequential(
            bn(width),
            relu(),
            conv3x3(width, width, stride=stride, groups=groups),
        )
        self.conv3 = nn.Sequential(
            bn(width),
            relu(),
            conv1x1(width, channels * self.expansion),
        )

        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.bn(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.bn(x)
            x = self.relu(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.sd(x) + skip

        return x


def conv1x1(in_channels, out_channels, stride=1, groups=1):
    return convnxn(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)


def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return convnxn(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, padding=1)


def convnxn(in_channels, out_channels, kernel_size, stride=1, groups=1, padding=0):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)


def relu():
    return nn.ReLU()


def relu6(mx=6.0, mn=0.0):
    return Lambda(lambda x: torch.clamp(x, mn, mx))


def bn(dim):
    return nn.BatchNorm2d(dim)


def ln1d(dim):
    return nn.LayerNorm(dim)


def ln2d(dim):
    return nn.Sequential(
        Rearrange("b c h w -> b h w c"),
        nn.LayerNorm(dim),
        Rearrange("b h w c -> b c h w"),
    )


def dense(in_features, out_features, bias=True):
    return nn.Linear(in_features, out_features, bias)


def blur(in_filters, sfilter=(1, 1), pad_mode="constant"):
    if tuple(sfilter) == (1, 1) and pad_mode in ["constant", "zero"]:
        layer = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
    else:
        layer = Blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)
    return layer


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class SamePad(nn.Module):

    def __init__(self, filter_size, pad_mode="constant", **kwargs):
        super(SamePad, self).__init__()

        self.pad_size = [
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
        ]
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, self.pad_size, mode=self.pad_mode)

        return x

    def extra_repr(self):
        return "pad_size=%s, pad_mode=%s" % (self.pad_size, self.pad_mode)


class Blur(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode="replicate", **kwargs):
        super(Blur, self).__init__()

        filter_size = len(sfilter)
        self.pad = SamePad(filter_size, pad_mode=pad_mode)

        self.filter_proto = torch.tensor(sfilter, dtype=torch.float, requires_grad=False)
        self.filter = torch.tensordot(self.filter_proto, self.filter_proto, dims=0)
        self.filter = self.filter / torch.sum(self.filter)
        self.filter = self.filter.repeat([in_filters, 1, 1, 1])
        self.filter = torch.nn.Parameter(self.filter, requires_grad=False)

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, self.filter, groups=x.size()[1])

        return x

    def extra_repr(self):
        return "pad=%s, filter_proto=%s" % (self.pad, self.filter_proto.tolist())


class Downsample(nn.Module):

    def __init__(self, strides=(2, 2), **kwargs):
        super(Downsample, self).__init__()

        if isinstance(strides, int):
            strides = (strides, strides)
        self.strides = strides

    def forward(self, x):
        shape = (-(-x.size()[2] // self.strides[0]), -(-x.size()[3] // self.strides[1]))
        x = F.interpolate(x, size=shape, mode='nearest')

        return x

    def extra_repr(self):
        return "strides=%s" % repr(self.strides)


class DropPath(nn.Module):

    def __init__(self, p, **kwargs):
        super().__init__()

        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)

        return x

    def extra_repr(self):
        return "p=%s" % repr(self.p)


class Lambda(nn.Module):

    def __init__(self, lmd):
        super().__init__()
        if not isinstance(lmd, types.LambdaType):
            raise Exception("`lmd` should be lambda ftn.")
        self.lmd = lmd

    def forward(self, x):
        return self.lmd(x)


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


# Attention Blocks

class AttentionBlockA(nn.Module):
    # Attention block with post-activation.
    # This block is for ablation study, and we do NOT use this block by default.
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
        if dim_in != dim_out * self.expansion:
            self.shortcut.append(conv1x1(dim_in, dim_out * self.expansion))
            self.shortcut.append(norm(dim_out * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.conv = nn.Sequential(
            conv1x1(dim_in, width, stride=stride),
            norm(width),
            activation(),
        )
        self.attn = attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm = norm(dim_out * self.expansion)
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.conv(x)
        x, attn = self.attn(x)
        x = self.norm(x)
        x = self.sd(x) + skip

        return x


class AttentionBasicBlockA(AttentionBlockA):
    expansion = 1


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


# Stems

class StemA(nn.Module):
    # Typical Stem stage for CNNs, e.g. ResNet or ResNeXt.
    # This block is for ablation study, and we do NOT use this block by default.

    def __init__(self, dim_in, dim_out, pool=True):
        super().__init__()

        self.layer0 = []
        if pool:
            self.layer0.append(convnxn(dim_in, dim_out, kernel_size=7, stride=2, padding=3))
            self.layer0.append(bn(dim_out))
            self.layer0.append(relu())
            self.layer0.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.layer0.append(conv3x3(dim_in, dim_out, stride=1))
            self.layer0.append(bn(dim_out))
            self.layer0.append(relu())
        self.layer0 = nn.Sequential(*self.layer0)

    def forward(self, x):
        x = self.layer0(x)
        return x


class StemB(nn.Module):
    # Stem stage for pre-activation pattern based on pre-activation ResNet.
    # We use this block by default.

    def __init__(self, dim_in, dim_out, pool=True):
        super().__init__()

        self.layer0 = []
        if pool:
            self.layer0.append(convnxn(dim_in, dim_out, kernel_size=3, stride=1, padding=1))
            self.layer0.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.layer0.append(conv3x3(dim_in, dim_out, stride=1))
        self.layer0 = nn.Sequential(*self.layer0)

    def forward(self, x):
        x = self.layer0(x)
        return x


# Model

class AlterNet(nn.Module):

    def __init__(self, block1, block2, *,
                 num_blocks, num_blocks2, heads,
                 cblock=BNGAPBlock,
                 sd=0.0, num_classes=10, stem=StemB, name="alternet", **block_kwargs):
        super().__init__()
        self.name = name
        idxs = [[j for j in range(sum(num_blocks[:i]), sum(num_blocks[:i + 1]))] for i in range(len(num_blocks))]
        sds = [[sd * j / (sum(num_blocks) - 1) for j in js] for js in idxs]

        self.layer0 = stem(3, 64)
        self.layer1 = self._make_layer(block1, block2, 64, 64,
                                       num_blocks[0], num_blocks2[0], stride=1, heads=heads[0], sds=sds[0], **block_kwargs)
        self.layer2 = self._make_layer(block1, block2, 64 * block2.expansion, 128,
                                       num_blocks[1], num_blocks2[1], stride=2, heads=heads[1], sds=sds[1], **block_kwargs)
        self.layer3 = self._make_layer(block1, block2, 128 * block2.expansion, 256,
                                       num_blocks[2], num_blocks2[2], stride=2, heads=heads[2], sds=sds[2], **block_kwargs)
        self.layer4 = self._make_layer(block1, block2, 256 * block2.expansion, 512,
                                       num_blocks[3], num_blocks2[3], stride=2, heads=heads[3], sds=sds[3], **block_kwargs)

        self.classifier = []
        if cblock is MLPBlock:
            self.classifier.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.classifier.append(cblock(7 * 7 * 512 * block2.expansion, num_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(512 * block2.expansion, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)

    @staticmethod
    def _make_layer(block1, block2, in_channels, out_channels, num_block1, num_block2, stride, heads, sds, **block_kwargs):
        alt_seq = [False] * (num_block1 - num_block2 * 2) + [False, True] * num_block2
        stride_seq = [stride] + [1] * (num_block1 - 1)

        seq, channels = [], in_channels
        for alt, stride, sd in zip(alt_seq, stride_seq, sds):
            block = block1 if not alt else block2
            seq.append(block(channels, out_channels, stride=stride, sd=sd, heads=heads, **block_kwargs))
            channels = out_channels * block.expansion

        return nn.Sequential(*seq)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x


def AlterNet18(conf, stem=True, name="alternet_18", **block_kwargs):
    return AlterNet(BasicBlock, AttentionBasicBlockB, stem=partial(StemB, pool=stem),
                    num_blocks=(2, 2, 2, 2), num_blocks2=(0, 1, 1, 1), heads=(3, 6, 12, 24),
                    num_classes=conf.emd_size, name=name, **block_kwargs)


def AlterNet34(conf, stem=True, name="alternet_34", **block_kwargs):
    return AlterNet(BasicBlock, AttentionBasicBlockB, stem=partial(StemB, pool=stem),
                    num_blocks=(3, 4, 6, 4), num_blocks2=(0, 1, 3, 2), heads=(3, 6, 12, 24),
                    num_classes=conf.emd_size, name=name, **block_kwargs)


def AlterNet50(conf, stem=True, name="alternet_50", **block_kwargs):
    return AlterNet(Bottleneck, AttentionBlockB, stem=partial(StemB, pool=stem),
                    num_blocks=(3, 4, 6, 4), num_blocks2=(0, 1, 3, 2), heads=(3, 6, 12, 24),
                    num_classes=conf.emd_size, name=name, **block_kwargs)


def AlterNet101(conf, stem=True, name="alternet_101", **block_kwargs):
    return AlterNet(Bottleneck, AttentionBlockB, stem=partial(StemB, pool=stem),
                    num_blocks=(3, 4, 23, 4), num_blocks2=(0, 1, 3, 2), heads=(3, 6, 12, 24),
                    num_classes=conf.emd_size, name=name, **block_kwargs)


def AlterNet152(conf, stem=True, name="alternet_152", **block_kwargs):
    return AlterNet(Bottleneck, AttentionBlockB, stem=partial(StemB, pool=stem),
                    num_blocks=(3, 8, 36, 4), num_blocks2=(0, 1, 3, 2), heads=(3, 6, 12, 24),
                    num_classes=conf.emd_size, name=name, **block_kwargs)


def Encoder(conf):
    if conf.network == 'AlterNet152':
        return AlterNet152(conf)
    elif conf.network == 'AlterNet101':
        return AlterNet101(conf)
    elif conf.network == 'AlterNet50':
        return AlterNet50(conf)
    elif conf.network == 'AlterNet34':
        return AlterNet34(conf)
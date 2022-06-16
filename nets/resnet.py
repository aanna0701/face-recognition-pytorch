import torch
import torch.nn as nn

# ======================================= Convolution layers =======================================

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

    This layer creates a basic residual block of ResNet architecture, which is a pre-activation Residual Unit.
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

    This layer creates a bottleneck residual block of ResNet architecture, which is a original Residual Unit.
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
# ResNet backbone
# --------------------------------------------
class ResNet(nn.Module):
    """ Implement of Residual network (ResNet) (https://arxiv.org/pdf/1512.03385.pdf):

    This layer creates a ResNet model by stacking basic or bottleneck residual blocks.

    Args:
        block: block to stack in each layer - BasicBlock or Bottleneck
        layers: # of stacked blocks in each layer
    """
    def __init__(self, block, layers, conf):
        # options from configuration
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = conv3x3(3, self.inplanes, stride=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.stack_layers(block, 64, layers[0])
        self.layer2 = self.stack_layers(block, 128, layers[1], stride=2)
        self.layer3 = self.stack_layers(block, 256, layers[2], stride=2)
        self.layer4 = self.stack_layers(block, conf.emd_size, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(block.expansion * conf.emd_size)
        self.dropout = nn.Dropout()
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

    def stack_layers(self, block, planes, blocks, stride=1):
        downsample = None
        # Peforms downsample if stride != 1 of inplanes != block.expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []
        # For the first residual block, stride is 1 or 2
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        # From the second residual block, stride is 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        
        return x

# --------------------------------------------
# Define ResNet models
# --------------------------------------------
def ResNet18(conf, **kwargs):
    """ Constructs a ResNet-18 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    model = ResNet(ResidualBlock, [2, 2, 2, 2], conf, **kwargs)

    
    return model

def ResNet34(conf, **kwargs):
    """ Constructs a ResNet-34 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    model = ResNet(ResidualBlock, [3, 4, 6, 4], conf, **kwargs)

    
    return model

def ResNet50(conf, **kwargs):
    """ Constructs a ResNet-50 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    model = ResNet(ResidualBlock, [3, 4, 14, 4], conf, **kwargs)

    
    return model

def ResNet100(conf, **kwargs):
    """ Constructs a ResNet-100 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    model = ResNet(ResidualBlock, [3, 13, 30, 4], conf, **kwargs)

    
    return model

def ResNet200(conf, **kwargs):
    """ Constructs a ResNet-100 model
    Args:
        conf: configurations
    """
    ResidualBlock = BasicBlock
    model = ResNet(ResidualBlock, [3, 43, 50, 4], conf, **kwargs)

    
    return model

def Encoder(conf):
    if conf.network == 'ResNet200':
        return ResNet200(conf)
    elif conf.network == 'ResNet100':
        return ResNet100(conf)
    elif conf.network == 'ResNet50':
        return ResNet50(conf)
    elif conf.network == 'ResNet34':
        return ResNet34(conf)
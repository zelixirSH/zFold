import torch
from torch import nn
import torch.nn.functional as F

Sequential = nn.Sequential

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def bn_conv1d(in_planes, out_planes, kernel_size, dilated, bias):
    return nn.Sequential(
        nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            dilation=dilated,
            padding=(dilated * (kernel_size - 1) + 1) // 2,
            bias=bias,
        ),
        nn.BatchNorm1d(out_planes),
    )

def in_conv1d(in_planes, out_planes, kernel_size, dilated, bias):
    return nn.Sequential(
        nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            dilation=dilated,
            padding=(dilated * (kernel_size - 1) + 1) // 2,
            bias=bias,
        ),
        nn.InstanceNorm1d(out_planes),
    )

def bn_conv2d(in_planes, out_planes, kernel_size, dilated, bias):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            dilation=dilated,
            padding=(dilated * (kernel_size - 1) + 1) // 2,
            bias=bias,
        ),
        nn.BatchNorm2d(out_planes),
    )

#================================================================================================================
class BottleNeck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,  dilation=1, norm_layer=None):
        super(BottleNeck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ELU(inplace=True)
        self.conv1 = conv1x1(inplanes, planes//2, stride)

        self.bn2 = norm_layer(planes//2)
        self.relu2 = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes//2, planes//2, stride, dilation=dilation)

        self.bn3 = norm_layer(inplanes//2)
        self.relu3 = nn.ELU(inplace=True)
        self.conv3 = conv1x1(inplanes//2, planes, stride)

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out += identity
        return out

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class SEBottleneck(nn.Module):
    """
    SEBottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1, norm_layer = nn.InstanceNorm2d,
                 is_depthwize = False):
        super(SEBottleneck, self).__init__()

        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.bn0 = norm_layer(inplanes)
            self.relu0 = nn.ELU(inplace=True)
            self.conv0 = conv1x1(inplanes, planes, stride)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(planes)
        self.bn2 = norm_layer(planes // 2)
        self.bn3 = norm_layer(planes // 2)

        self.relu1 = nn.ELU(inplace=True)
        self.conv1 = conv1x1(planes, planes//2, stride)

        self.relu2 = nn.ELU(inplace=True)
        if is_depthwize:
            self.conv2 = conv3x3(planes//2, planes//2, stride, groups=planes//2, dilation=dilation)
        else:
            self.conv2 = conv3x3(planes//2, planes//2, stride, groups=groups, dilation=dilation)

        self.relu3 = nn.ELU(inplace=True)
        self.conv3 = conv1x1(planes//2, planes, stride)
        self.se_module = SEModule(planes, reduction=16)
        self.stride = stride

    def forward(self, x):
        if self.inplanes != self.planes:
            x = self.bn0(x)
            x = self.relu0(x)
            x = self.conv0(x)

        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out = self.se_module(out) + identity
        return out

class SEBottleneckInvolution(nn.Module):
    """
    SEBottleneck
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1, norm_layer=None):
        super(SEBottleneckInvolution, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ELU(inplace=True)
        self.conv1 = conv1x1(inplanes, planes//2, stride)

        self.bn2 = norm_layer(planes//2)
        self.relu2 = nn.ELU(inplace=True)
        self.conv2 = involution(channels=planes//2, kernel_size=11, stride=1)

        self.bn3 = norm_layer(inplanes//2)
        self.relu3 = nn.ELU(inplace=True)
        self.conv3 = conv1x1(inplanes//2, planes, stride)
        self.se_module = SEModule(planes, reduction=16)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out = self.se_module(out) + identity
        return out

class DeformSEBottleneck(nn.Module):
    """
    SEBottleneck
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1,  dilation=1, norm_layer=None):
        super(DeformSEBottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ELU(inplace=True)
        self.conv1 = conv1x1(inplanes, planes//2, stride)

        self.bn2 = norm_layer(planes//2)
        self.relu2 = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes//2, planes//2, stride, dilation=dilation, is_deform=True)

        self.bn3 = norm_layer(inplanes//2)
        self.relu3 = nn.ELU(inplace=True)
        self.conv3 = conv1x1(inplanes//2, planes, stride)
        self.se_module = SEModule(planes, reduction=16)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out = self.se_module(out) + identity
        return out

class SEDilatedBottleneck(nn.Module):
    """
    SEDilatedBottleneck
    """
    def __init__(self, inplanes, planes, stride=1,  dilation=1, norm_layer=None):
        super(SEDilatedBottleneck, self).__init__()

        dplanes = planes // 4
        self.b1 = SEBottleneck(inplanes, dplanes, stride=1,  dilation=1)
        self.b2 = SEBottleneck(inplanes, dplanes, stride=1,  dilation=2)
        self.b4 = SEBottleneck(inplanes, dplanes, stride=1,  dilation=4)
        self.b8 = SEBottleneck(inplanes, dplanes, stride=1,  dilation=8)

    def forward(self, x):

        b1 = self.b1(x)
        b2 = self.b2(x)
        b4 = self.b4(x)
        b8 = self.b8(x)

        out = torch.cat([b1, b2, b4, b8], dim=1)
        return out

if __name__ == '__main__':
    import numpy as np
    planes = 64
    norm_layer = nn.InstanceNorm2d
    model = SEBottleneck(planes, planes, stride=1, dilation=1, is_depthwize=True, norm_layer=norm_layer)
    print(model)

    inputs = torch.FloatTensor(np.zeros([64,64,32,32]))
    outputs = model(inputs)
    print(outputs.shape)




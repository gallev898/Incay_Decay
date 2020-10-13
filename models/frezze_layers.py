import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut = nn.Sequential(
                conv,
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EmbeddingResNet(nn.Module):
    def __init__(self, block, num_blocks, embedding_size, num_classes=10):
        super(EmbeddingResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.fc_l1 = nn.Linear(4096,512)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.fc_l2 = nn.Linear(2048,512)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.fc_l3 = nn.Linear(1024,512)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, exit_layer):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out1 = self.layer1(out)
        if exit_layer == 'layer1':
            out = F.avg_pool2d(out1, 4)
            out_pre_norm = self.fc_l1(out.view(out.size(0), -1))
            out = F.normalize(out_pre_norm, p=2, dim=1)
            return out, out_pre_norm

        out2 = self.layer2(out1)
        if exit_layer == 'layer2':
            out = F.avg_pool2d(out2, 4)
            out_pre_norm = self.fc_l2(out.view(out.size(0), -1))
            out = F.normalize(out_pre_norm, p=2, dim=1)
            return out, out_pre_norm

        out3 = self.layer3(out2)
        if exit_layer == 'layer3':
            out = F.avg_pool2d(out3, 4)
            out_pre_norm = self.fc_l3(out.view(out.size(0), -1))
            out = F.normalize(out_pre_norm, p=2, dim=1)
            return out, out_pre_norm

        out4 = self.layer4(out3)
        if exit_layer == 'layer4':
            out = F.avg_pool2d(out4, 4)
            out_pre_norm = out.view(out.size(0), -1)
            out = F.normalize(out_pre_norm, p=2, dim=1)

        return out, out_pre_norm


def resnet18_sim_soft_freeze_layers(embedding_size):
    print('File Runing: freeze_layers.py')
    return EmbeddingResNet(BasicBlock, [2,2,2,2], embedding_size)

# frezze_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SE1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, hidden, 1, bias=True)
        self.fc2 = nn.Conv1d(hidden, channels, 1, bias=True)

    def forward(self, x):
        w = self.pool(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class ResidualBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, k=5, p=2, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=p, bias=False)
        self.gn1   = nn.GroupNorm(4, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, stride=1, padding=p, bias=False)
        self.gn2   = nn.GroupNorm(4, out_ch)
        self.se    = SE1D(out_ch, reduction=reduction)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        res = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            res = self.downsample(x)
        out = self.relu(out + res)
        return out

class ResNet1D_Nano_Final(nn.Module):
    def __init__(self, num_classes=7, input_dim=39, base=12, k1=7, k=5, reduction=8, coarse_classes=4):
        super().__init__()
        self.inplanes = base

        # 39 parametre: feature importance gate
        self.feature_gate = nn.Parameter(torch.zeros(1, 1, input_dim))

        self.conv1 = nn.Conv1d(1, base, kernel_size=k1, stride=1, padding=k1//2, bias=False)
        self.gn1   = nn.GroupNorm(4, base)
        self.relu  = nn.ReLU(inplace=True)

        self.reduction = reduction
        self.layer1 = self._make_layer(base,   stride=1, k=k)
        self.layer2 = self._make_layer(base*2, stride=2, k=k)
        self.layer3 = self._make_layer(base*4, stride=2, k=k)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.2)

        self.fc_main   = nn.Linear(base*4, num_classes)
        self.fc_coarse = nn.Linear(base*4, coarse_classes)

    def _make_layer(self, planes, stride=1, k=5):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(4, planes),
            )
        p = k // 2
        block = ResidualBlockSE(self.inplanes, planes, stride=stride, downsample=downsample, k=k, p=p, reduction=self.reduction)
        self.inplanes = planes
        return nn.Sequential(block)

    def forward(self, x, return_coarse=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # feature gate
        x = x * torch.sigmoid(self.feature_gate)

        x = self.relu(self.gn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)

        main = self.fc_main(x)
        if return_coarse:
            coarse = self.fc_coarse(x)
            return main, coarse
        return main

def create_ids_model(mode="multiclass", num_classes=7, input_dim=39):
    print("[FACTORY] ResNet1D-Nano Final (base=12, SE+Gate+CoarseHead) init")
    return ResNet1D_Nano_Final(num_classes=num_classes, input_dim=input_dim, base=12)

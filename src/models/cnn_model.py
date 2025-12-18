"""
SE-ResNet1D - ELITE EDITION
Target: >90% Accuracy on Attack-Only Dataset
Technique: Squeeze-and-Excitation (Channel Attention)
"""
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlockSE, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(4, out_channels)
        
        # ELITE ADDITION: SE Block
        self.se = SEBlock(out_channels)
        
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Dikkat Mekanizması Devrede
        out = self.se(out)
        
        if self.downsample:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out

class ResNet1D_Elite(nn.Module):
    def __init__(self, num_classes=7, input_dim=39):
        super(ResNet1D_Elite, self).__init__()
        self.inplanes = 32
        
        # Giriş Genişletme (Daha fazla özellik yakala)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(4, 32)
        self.relu = nn.ReLU(inplace=True)
        
        # 3 Katmanlı Güçlü Yapı
        self.layer1 = self._make_layer(32, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(4, planes),
            )
        layers = []
        layers.append(ResidualBlockSE(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlockSE(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def create_ids_model(mode: str = "multiclass", num_classes: int = 7, input_dim: int = 39):
    print(f"[FACTORY] Initializing ResNet1D-Elite (SE-Attention + Focal Ready).")
    model = ResNet1D_Elite(num_classes=num_classes, input_dim=input_dim)
    return model
"""
FR Backbone Models
Implements IR152 (ResNet-152 style) and IRSE50 (SE-ResNet-50 style)
used as surrogate and victim models in DPA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Basic building blocks
# ──────────────────────────────────────────────

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Conv2d 1x1 instead of Linear — matches checkpoint exactly
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)           # (b, c, 1, 1)
        y = torch.relu(self.fc1(y))    # (b, c//r, 1, 1)
        y = torch.sigmoid(self.fc2(y)) # (b, c, 1, 1)
        return x * y
    

class IRBlock_evoLVe(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super().__init__()

        if use_se:
            self.res_layer = nn.Sequential(
                nn.BatchNorm2d(in_channels),                               # index 0
                nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                          padding=1, bias=False),                          # index 1 ← stride here
                nn.PReLU(out_channels),                                    # index 2
                nn.Conv2d(out_channels, out_channels, 3, stride=1,
                          padding=1, bias=False),                          # index 3 ← stride=1
                nn.BatchNorm2d(out_channels),                              # index 4
                SEBlock(out_channels),                                     # index 5
            )
        else:
            self.res_layer = nn.Sequential(
                nn.BatchNorm2d(in_channels),                               # index 0
                nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                          padding=1, bias=False),                          # index 1 ← stride here
                nn.PReLU(out_channels),                                    # index 2
                nn.Conv2d(out_channels, out_channels, 3, stride=1,
                          padding=1, bias=False),                          # index 3 ← stride=1
                nn.BatchNorm2d(out_channels),                              # index 4
            )

        if stride != 1 or in_channels != out_channels:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut_layer = None

    def forward(self, x):
        res = self.res_layer(x)
        shortcut = self.shortcut_layer(x) if self.shortcut_layer else x
        return res + shortcut    
# ──────────────────────────────────────────────
# Generic backbone builder
# ──────────────────────────────────────────────

class IRBackbone(nn.Module):
    def __init__(self, input_size=(112,112), layers=(3,4,14,3),
                 use_se=False, emb_size=512):
        super().__init__()

        self.input_layer = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),  # index 0
                nn.BatchNorm2d(64),                                      # index 1
                nn.PReLU(64)                                             # index 2
            )

        blocks = []

# blocks 0-2: 64→64, stride=1
        for _ in range(layers[0]):
            blocks.append(IRBlock_evoLVe(64, 64, stride=1, use_se=use_se))

        # block 3: 64→128, stride=2
        blocks.append(IRBlock_evoLVe(64, 128, stride=2, use_se=use_se))
        # blocks 4-6: 128→128, stride=1
        for _ in range(layers[1] - 1):
            blocks.append(IRBlock_evoLVe(128, 128, stride=1, use_se=use_se))

        # block 7: 128→256, stride=2
        blocks.append(IRBlock_evoLVe(128, 256, stride=2, use_se=use_se))
        # blocks 8-20: 256→256, stride=1
        for _ in range(layers[2] - 1):

            blocks.append(IRBlock_evoLVe(256, 256, stride=1, use_se=use_se))

        # block 21: 256→512, stride=2
        blocks.append(IRBlock_evoLVe(256, 512, stride=2, use_se=use_se))
        # blocks 22-23: 512→512, stride=1
        for _ in range(layers[3] - 1):
            blocks.append(IRBlock_evoLVe(512, 512, stride=1, use_se=use_se))

        self.body = nn.Sequential(*blocks)

        fc_in = 512 * 7 * 7

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.4),
            Flatten(),
            nn.Linear(512 * 7 * 7, emb_size),
            nn.BatchNorm1d(emb_size)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
    # Force BN to use batch statistics to avoid running stats collapse
        for module in self.body.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()
        
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x
# ──────────────────────────────────────────────
# Specific architecture constructors
# ──────────────────────────────────────────────

def IR_50(input_size=(112, 112), emb_size=512):
    """IR-50: used as base for IRSE50 when use_se=False."""
    return IRBackbone(input_size=input_size, layers=[3, 4, 14, 3],
                      use_se=False, emb_size=emb_size)


def IR_152(input_size=(112, 112), emb_size=512):
    """IR-152: deepest standard backbone in DPA experiments."""
    return IRBackbone(input_size=input_size, layers=[3, 8, 36, 3],
                      use_se=False, emb_size=emb_size)


def IRSE_50(input_size=(112, 112), emb_size=512):
    """IRSE-50: IR-50 with Squeeze-and-Excitation."""
    return IRBackbone(input_size=input_size, layers=[3, 4, 14, 3],
                      use_se=True, emb_size=emb_size)


# ──────────────────────────────────────────────
# FaceNet-style lightweight backbone (MobileNet-V1 inspired)
# ──────────────────────────────────────────────

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)
        
        self.conv = nn.Sequential(
            # expand: conv.0
            nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),  # conv.0.0
                nn.BatchNorm2d(mid_ch),                    # conv.0.1
            ),
            # depthwise: conv.1
            nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                          padding=1, groups=mid_ch, bias=False),  # conv.1.0
                nn.BatchNorm2d(mid_ch),                           # conv.1.1
            ),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),  # conv.2
            nn.BatchNorm2d(out_ch),                     # conv.3
        )

    def forward(self, x):
        # if self.use_res:
        #     return x + self.conv(x)
        return self.conv(x)   # ← NO residual for most blocks
    
class DepthwiseConv(nn.Module):
    """Initial dw_conv block."""
    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False)
        self.pointwise = nn.Conv2d(64, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return x

class GlobalDepthwiseConv(nn.Module):
    """gdconv block — global depthwise, input must be 7×7."""
    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(512, 512, 7, groups=512, bias=False, padding=0)
        self.bn = nn.BatchNorm2d(512)

    def forward(self, x):
        # Safety check
        if x.shape[-1] != 7:
            raise RuntimeError(f"gdconv expects 7×7 input, got {x.shape[-1]}×{x.shape[-1]}")
        x = self.depthwise(x)
        x = self.bn(x)
        return x    

class MobileFaceNet(nn.Module):
    """
    MobileFaceNet (Sheng Chen et al. 2018) matching pretrained checkpoint exactly.
    """
    def __init__(self, emb_size=512):
        super().__init__()

        # Initial conv: conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),  # conv1.0
            nn.BatchNorm2d(64),                                      # conv1.1
        )

        # Depthwise block: dw_conv
        self.dw_conv = DepthwiseConv()

        # Expansion conv: conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 512, 1, bias=False),   # conv2.0
            nn.BatchNorm2d(512),                   # conv2.1
        )

        # Global depthwise: gdconv
        self.gdconv = GlobalDepthwiseConv()

        # Projection: conv3 + bn
        self.conv3 = nn.Conv2d(512, 128, 1, bias=True)
        self.bn    = nn.BatchNorm2d(128)

        # Inverted residual blocks: features.0 - features.14
        # Inferred from checkpoint channel sizes
        self.features = nn.Sequential(
            InvertedResidual(64,  64,  stride=1, expand_ratio=2),   # 0
            InvertedResidual(64,  64,  stride=1, expand_ratio=2),   # 1
            InvertedResidual(64,  64,  stride=1, expand_ratio=2),   # 2
            InvertedResidual(64,  64,  stride=1, expand_ratio=2),   # 3
            InvertedResidual(64,  64,  stride=1, expand_ratio=2),   # 4
            InvertedResidual(64,  128, stride=2, expand_ratio=4),   # 5
            InvertedResidual(128, 128, stride=1, expand_ratio=2),   # 6
            InvertedResidual(128, 128, stride=1, expand_ratio=2),   # 7
            InvertedResidual(128, 128, stride=1, expand_ratio=2),   # 8
            InvertedResidual(128, 128, stride=1, expand_ratio=2),   # 9
            InvertedResidual(128, 128, stride=1, expand_ratio=2),   # 10
            InvertedResidual(128, 128, stride=1, expand_ratio=2),   # 11
            InvertedResidual(128, 128, stride=2, expand_ratio=4),   # 12
            InvertedResidual(128, 128, stride=2, expand_ratio=2),   # 13
            InvertedResidual(128, 128, stride=1, expand_ratio=2),   # 14
        )

        self.flatten = Flatten()
        self.fc      = nn.Linear(128, emb_size)

    def forward(self, x):
    # Force all BN layers to use batch statistics
    # (running stats from checkpoint are miscalibrated)
        for module in self.features.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()
    
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.features(x)
        x = self.conv2(x)
        x = self.gdconv(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# ──────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────

MODEL_REGISTRY = {
    'IR_50':       IR_50,
    'IR_152':      IR_152,
    'IRSE_50':     IRSE_50,
    'MobileFace':  MobileFaceNet,
    'MobileFaceNet': MobileFaceNet,
}


def get_model(name: str, input_size=(112, 112), emb_size=512) -> nn.Module:
    """Factory function to instantiate a backbone by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    if name == 'MobileFace':
        return MODEL_REGISTRY[name](emb_size=emb_size)
    return MODEL_REGISTRY[name](input_size=input_size, emb_size=emb_size)
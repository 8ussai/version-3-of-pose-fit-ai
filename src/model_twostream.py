import torch
import torch.nn as nn
import torchvision.models as tvm

class FrameEncoder(nn.Module):
    """
    Encoder لكل فريم (CNN) ثم mean pooling زمنياً
    Input: (B,T,3,H,W)
    Output: (B,out_dim)
    """
    def __init__(self, out_dim=256, pretrained=False):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = tvm.MobileNet_V3_Small_Weights.DEFAULT
            except Exception:
                weights = None

        base = tvm.mobilenet_v3_small(weights=weights)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # mobilenet_v3_small channels = 576
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # x: (B,T,3,H,W)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.backbone(x)
        x = self.pool(x)         # (B*T,576,1,1)
        x = self.proj(x)         # (B*T,out_dim)
        x = x.reshape(b, t, -1)  # (B,T,out_dim)
        x = x.mean(dim=1)        # (B,out_dim)
        return x

class PoseEncoder(nn.Module):
    """
    GRU على تسلسل الـpose ثم mean pooling زمنياً (بدل آخر خطوة).
    Input: (B,T,132)
    Output: (B,out_dim)
    """
    def __init__(self, in_dim=132, hidden=128, out_dim=128, num_layers=1, dropout=0.1, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        gru_out_dim = hidden * (2 if bidirectional else 1)

        self.proj = nn.Sequential(
            nn.Linear(gru_out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, p):
        # p: (B,T,132)
        y, _ = self.gru(p)      # y: (B,T,H*(2 if bi else 1))
        y = y.mean(dim=1)       # mean على الزمن -> (B, *)
        y = self.proj(y)        # (B,out_dim)
        return y

class TwoStreamFusionNet(nn.Module):
    """
    Stream 1: RGB frames encoder
    Stream 2: Pose GRU encoder
    Fusion: concat -> head
    """
    def __init__(self, num_classes: int, pretrained_backbone: bool = False):
        super().__init__()
        self.rgb = FrameEncoder(out_dim=256, pretrained=pretrained_backbone)
        self.pose = PoseEncoder(in_dim=132, hidden=128, out_dim=128, num_layers=1, dropout=0.1, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames, pose):
        a = self.rgb(frames)
        b = self.pose(pose)
        x = torch.cat([a, b], dim=1)
        return self.head(x)  # logits

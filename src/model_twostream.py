import torch
import torch.nn as nn
import torchvision.models as tvm

class FrameEncoder(nn.Module):
    """
    Encoder لكل فريم (2D CNN خفيف) ثم تجميع زمني (mean)
    """
    def __init__(self, out_dim=256):
        super().__init__()
        base = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # x: (B,T,3,H,W)
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        f = self.backbone(x)
        f = self.pool(f)            # (B*T, 576,1,1)
        f = self.proj(f)            # (B*T, out_dim)
        f = f.reshape(B, T, -1)     # (B,T,out_dim)
        f = f.mean(dim=1)           # (B,out_dim)
        return f

class PoseEncoder(nn.Module):
    """
    Pose features (T,132) -> GRU -> embedding
    """
    def __init__(self, in_dim=132, hidden=128, out_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden*2, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, p):
        # p: (B,T,132)
        y, _ = self.gru(p)
        # خذ آخر خطوة زمنية
        last = y[:, -1, :]
        return self.proj(last)

class TwoStreamFusionNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.rgb = FrameEncoder(out_dim=256)
        self.pose = PoseEncoder(in_dim=132, hidden=128, out_dim=128)

        self.head = nn.Sequential(
            nn.Linear(256+128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames, pose):
        a = self.rgb(frames)
        b = self.pose(pose)
        x = torch.cat([a, b], dim=1)
        return self.head(x)

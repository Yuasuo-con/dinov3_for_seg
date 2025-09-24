import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- primitives ----
class DWConvBlock(nn.Module):
    """Depthwise separable conv block: DWConv -> PWConv -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, norm=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride=1, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class UpFuseBlock(nn.Module):
    """Upsample top_feat to lateral size and fuse: Conv1x1(reduce) + upsample + lateral 1x1 + DWConv"""
    def __init__(self, top_ch, lateral_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(top_ch, out_ch, 1, bias=False)
        self.lateral = nn.Conv2d(lateral_ch, out_ch, 1, bias=False)
        self.conv = DWConvBlock(out_ch, out_ch)
    def forward(self, top_feat, lateral_feat):
        top_up = F.interpolate(self.reduce(top_feat), size=lateral_feat.shape[-2:], mode='bilinear', align_corners=False)
        lat = self.lateral(lateral_feat)
        x = top_up + lat
        x = self.conv(x)
        return x

# ---- Light Segmentation Head ----
class LightSegHead(nn.Module):
    def __init__(self, in_channels=768, mid_channels=128, num_classes=3, use_aux=False):
        """
        in_channels: adapter embed dim (e.g., 384)
        mid_channels: internal FPN channel (128 recommended)
        num_classes: number of classes (1 for binary)
        use_aux: whether to provide auxiliary logits at intermediate scales
        """
        super().__init__()
        self.mid = mid_channels
        # lateral 1x1 to unify channels
        self.lateral4 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)  # f4 (1/32)
        self.lateral3 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)  # f3 (1/16)
        self.lateral2 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)  # f2 (1/8)
        self.lateral1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)  # f1 (1/4)

        # up & fuse blocks
        self.up43 = UpFuseBlock(mid_channels, mid_channels, mid_channels)  # f4 -> f3
        self.up32 = UpFuseBlock(mid_channels, mid_channels, mid_channels)  # -> f2
        self.up21 = UpFuseBlock(mid_channels, mid_channels, mid_channels)  # -> f1

        # refine convs
        self.refine = nn.Sequential(
            DWConvBlock(mid_channels, mid_channels),
            DWConvBlock(mid_channels, mid_channels),
        )
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

        self.use_aux = use_aux
        if self.use_aux:
            self.aux3 = nn.Conv2d(mid_channels, num_classes, 1)
            self.aux2 = nn.Conv2d(mid_channels, num_classes, 1)

        # init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats: dict):
        # feats: "1","2","3","4"
        f1 = feats["1"]  # (B,C,4H,4W)  -- finest spatial
        f2 = feats["2"]  # (B,C,2H,2W)
        f3 = feats["3"]  # (B,C,H, W)
        f4 = feats["4"]  # (B,C,H/2,W/2)

        # lateral projections
        r4 = self.lateral4(f4)
        r3 = self.lateral3(f3)
        r2 = self.lateral2(f2)
        r1 = self.lateral1(f1)

        # up & fuse: f4->f3, f3->f2, f2->f1
        u3 = self.up43(r4, r3)   # -> spatial of r3
        u2 = self.up32(u3, r2)   # -> spatial of r2
        u1 = self.up21(u2, r1)   # -> spatial of r1

        x = self.refine(u1)      # refine at finest
        logits = self.classifier(x)  # B x num_classes x (H1) x (W1)

        out = {"logits": logits}
        if self.use_aux:
            out["aux3"] = F.interpolate(self.aux3(u3), size=logits.shape[-2:], mode='bilinear', align_corners=False)
            out["aux2"] = F.interpolate(self.aux2(u2), size=logits.shape[-2:], mode='bilinear', align_corners=False)
        return out["logits"]


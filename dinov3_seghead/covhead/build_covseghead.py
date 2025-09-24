
from functools import partial

import torch
from torch.nn import functional as F

from dinov3_seghead.backbone.dinov3_adapter import DINOv3_Adapter
from dinov3_seghead.covhead.light_seghead import LightSegHead


BACKBONE_INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}

class FeatureDecoder(torch.nn.Module):
    def __init__(self, segmentation_model: torch.nn.ModuleList):
        super().__init__()
        self.segmentation_model = segmentation_model

    def forward(self, inputs):
        # backbone forward
        feats = self.segmentation_model[0](inputs)
        # decoder head
        out = self.segmentation_model[1](feats)
        
        seg = F.interpolate(out, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        return seg   # [B, C, H, W]



def build_light_seg_decoder(
    backbone_model,
    backbone_name,
    num_classes=4,
):

    backbone_model = DINOv3_Adapter(
        backbone_model,
        interaction_indexes=BACKBONE_INTERMEDIATE_LAYERS[backbone_name],
    )
    backbone_model.eval()
    embed_dim = backbone_model.backbone.embed_dim
    patch_size = backbone_model.patch_size
    decoder = LightSegHead(in_channels=embed_dim, num_classes=num_classes)

    segmentation_model = FeatureDecoder(
        torch.nn.ModuleList(
            [
                backbone_model,
                decoder,
            ]
        ),
    )
    return segmentation_model

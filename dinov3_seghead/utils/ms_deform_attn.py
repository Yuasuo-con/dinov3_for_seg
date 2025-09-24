# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd

from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MSDA
except ImportError:
    # if we just care about inference, we don't need
    # the compiled extension for multi-scale deformable attention
    MSDA = None


class MSDeformAttnFunction(Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
    ):
        ctx.im2col_step = im2col_step
        output = ms_deform_attn_core_pytorch(
            value,
            value_spatial_shapes,
            #  value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        ctx.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if MSDA is None:
            raise RuntimeError(
                "MultiScaleDeformableAttention is not available, "
                "please compile with CUDA if you want to train a "
                "segmentation head with deformable attention"
            )
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    """
    Wrapper for MMCV's MultiScaleDeformableAttention.
    Keeps interface consistent with Deformable DETR / Mask2Former.
    """

    def __init__(self, embed_dim=256, num_levels=4, num_heads=8, num_points=4):
        """
        Args:
            embed_dim (int): feature dimension
            num_levels (int): number of feature levels (multi-scale)
            num_heads (int): number of attention heads
            num_points (int): number of sampling points per head per level
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        # directly use mmcv implementation
        self.attn = MSDA(
            embed_dims=embed_dim,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
            batch_first=True
        )

    def forward(self, query, reference_points, value, spatial_shapes, level_start_index, padding_mask=None):
        """
        Args:
            query (Tensor): [B, Len_q, C]
            reference_points (Tensor): [B, Len_q, num_levels, 2], normalized coords
            value (Tensor): [B, Σ(HiWi), C], flattened multi-scale feature maps
            spatial_shapes (Tensor): [num_levels, 2], each (Hi, Wi)
            level_start_index (Tensor): [num_levels], starting index of each level
            padding_mask (Tensor, optional): [B, Σ(HiWi)]
        Returns:
            Tensor: [B, Len_q, C]
        """
        return self.attn(query=query, reference_points=reference_points, value=value, spatial_shapes=spatial_shapes, level_start_index=level_start_index)
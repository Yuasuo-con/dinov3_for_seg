#!/usr/bin/env python3
# test_multi_case.py
"""
Single-GPU multi-case testing script.

Usage:
    python test_multi_case.py \
        --data_root /path/to/dataset \
        --val_list val_list.txt \
        --model_name covhead \
        --weights /path/to/ckpt.pth \
        --save_pred_dir ./preds \
        --num_classes 4

Assumptions:
 - dataset layout:
     dataset/
       images/
       masks/
       val_list.txt (one case name per line, no extension)
 - CT2DDataset returns per-case groups:
     __getitem__(i) -> (imgs_groups, masks_groups)
     imgs_groups: Tensor [G, b, 3, H, W]
     masks_groups: Tensor [G, b, H, W]
 - You will supply eval_utils.eval_case_metrics(pred_volume, gt_volume, num_classes)
   which returns a dict like {'dice':..., 'iou':..., 'asd':..., 'hd95':...}
"""

import os
import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

# import your dataset and model builders (same as train.py)
from data.datasets import CT2DDataset
from models.build_vit import dinov3_vitb16, dinov3_vitl16
from models import UNet
from dinov3_seghead.build_seghead import build_segmentation_decoder
from dinov3_seghead.covhead.build_covseghead import build_light_seg_decoder


# -------------------------
# helpers
# -------------------------
def init_logger(log_file: str = None):
    logger = logging.getLogger("test_multi_case")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # file
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix if present (checkpoint saved from DDP)."""
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module.") :]
        new_state[new_k] = v
    return new_state


def load_checkpoint_to_model(model: torch.nn.Module, ckpt_path: str, logger=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    state = strip_module_prefix(state)
    model.load_state_dict(state)
    if logger:
        logger.info(f"Loaded checkpoint: {ckpt_path}")


def save_nifti_volume(volume_np: np.ndarray, reference_nii_path: str, out_path: str):
    """
    volume_np: [D, H, W] (numeric)
    reference_nii_path: used to copy spacing/origin/direction
    """
    img = sitk.GetImageFromArray(volume_np.astype(np.uint8))
    ref = sitk.ReadImage(reference_nii_path)
    img.SetSpacing(ref.GetSpacing())
    img.SetOrigin(ref.GetOrigin())
    img.SetDirection(ref.GetDirection())
    sitk.WriteImage(img, out_path)


# -------------------------
# main testing routine
# -------------------------
def test_all_cases(
    model: torch.nn.Module,
    dataset: CT2DDataset,
    device: torch.device,
    eval_fn,
    save_pred_dir: str = None,
    logger: logging.Logger = None,
):
    """
    Iterate dataset case-by-case (dataset.__getitem__(i) returns a case),
    run inference and call eval_fn(pred_volume, gt_volume, num_classes)

    Returns:
        per_case_results: list of dicts, each has case_name and metrics
    """
    model.eval()
    per_case_results = []

    # Ensure save dir exists
    if save_pred_dir:
        os.makedirs(save_pred_dir, exist_ok=True)

    # iterate by index to know case name
    for idx in range(len(dataset)):
        case_name = dataset.case_list[idx] if hasattr(dataset, "case_list") else f"case_{idx}"
        case_path = os.path.join(dataset.image_dir, case_name)
        # get spacing
        spacing = sitk.ReadImage(case_path).GetSpacing()[::-1]

        if logger:
            logger.info(f"Processing case {idx+1}/{len(dataset)}: {case_name}")

        imgs_groups, masks_groups = dataset[idx]  # imgs_groups: [G, b, 3, H, W], masks_groups: [G, b, H, W]
        # ensure numpy tensors
        assert isinstance(imgs_groups, torch.Tensor) and isinstance(masks_groups, torch.Tensor), \
            "Dataset must return torch.Tensor groups"

        G, b, C, H, W = imgs_groups.shape
        preds_slices = []

        with torch.no_grad():
            for g in range(G):
                imgs = imgs_groups[g]  # [b, 3, H, W]
                # to device & proper dtype
                imgs = imgs.to(device=device, dtype=torch.float32)
                logits = model(imgs)  # assume [b, num_classes, H, W]
                # convert to predictions
                if logits.shape[1] == 1:
                    # binary: threshold 0.5
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long().squeeze(1).cpu().numpy()  # [b, H, W]
                else:
                    probs = F.softmax(logits, dim=1)
                    preds = probs.argmax(dim=1).cpu().numpy()  # [b, H, W]
                preds_slices.append(preds)

        # concat along depth
        preds_volume_padded = np.concatenate(preds_slices, axis=0)  # [D_padded, H, W]

        # read original gt to get true depth (and also for evaluation)
        # dataset is expected to have mask_dir and case_list
        # prefer reading from dataset.mask_dir to be robust
        if hasattr(dataset, "mask_dir"):
            mask_case_name = '3classesmask_case' + re.search(r"Case(\d+)\.nii\.gz", case_name).group(1) + '.nii.gz'
            mask_path = os.path.join(dataset.mask_dir, mask_case_name)
        else:
            # fallback: assume dataset.root + /masks/
            mask_path = os.path.join(dataset.root, "masks", case_name)
        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.int32)  # [D, H, W]
        D_true = gt_arr.shape[0]

        # crop padded preds to original depth
        preds_volume = preds_volume_padded[:D_true]

        # compute metrics for this case by calling eval_fn (provided by user)
        # expected signature: eval_fn(pred_volume: np.ndarray [D,H,W], gt_volume: np.ndarray [D,H,W], num_classes: int) -> dict
        metrics: Dict[str, Any] = eval_fn(pred_volume=preds_volume, gt_volume=gt_arr, spacing=spacing, num_classes=dataset.num_classes if hasattr(dataset, "num_classes") else None)
        # metrics should be dict, e.g. {'dice': val, 'iou': val, 'asd': val, 'hd95': val}
        result = {"case": case_name}
        result.update(metrics)
        per_case_results.append(result)

        # optionally save predicted nii
        if save_pred_dir:
            out_file = os.path.join(save_pred_dir, f"{case_name.split('.')[0]}_pred.nii.gz")
            # use image as reference for spacing/origin/direction if available
            if hasattr(dataset, "image_dir"):
                ref_img_path = os.path.join(dataset.image_dir, case_name)
                save_nifti_volume(preds_volume, ref_img_path, out_file)
            else:
                # write without reference
                sitk.WriteImage(sitk.GetImageFromArray(preds_volume.astype(np.uint8)), out_file)

        if logger:
            logger.info(f"Case {case_name} metrics: {metrics}")

    # aggregate
    # find metric keys (exclude 'case')
    all_keys = [k for k in per_case_results[0].keys() if k != "case"]
    summary = {}
    for k in all_keys:
        vals = [r[k] for r in per_case_results]
        vals = np.array(vals, dtype=float)
        # flatten
        vals = vals.flatten()
        summary[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "n": len(vals) / dataset.num_classes}

    if logger:
        logger.info("==== Summary ====")
        for k, v in summary.items():
            logger.info(f"{k}: mean={v['mean']:.4f} std={v['std']:.4f} n={v['n']}")

    return per_case_results, summary


# -------------------------
# CLI & main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/root/data/public/Spine_qilu/ori", help="dataset root (images/ masks/)")
    p.add_argument("--val_list", default='./test_list.txt',help="val list file (one case per line)")
    p.add_argument("--backbone", choices=["vitb16", "vitl16", "unet"], default="vitl16", help="backbone name, if unet, seghead is ignored")
    p.add_argument("--seghead", choices=["m2f", "covhead", ''], default="covhead", help="segmentation head name")
    p.add_argument("--weights", default="/root/data/private/dinov3_workdir/fewshot_vitl16_lightseg_4case/ckpt_epoch47.pth",help="checkpoint path (state dict or checkpoint with 'model_state')")
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--slice_batch_size", type=int, default=32)
    p.add_argument("--case_batch_size", type=int, default=1, help="cases per iteration; recommend 1 for per-case metrics")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--hu_min", type=int, default=-1000)
    p.add_argument("--hu_max", type=int, default=400)
    p.add_argument("--norm", choices=["zscore", "minmax", "none"], default="zscore")
    p.add_argument("--save_pred_dir", default="/root/data/private/dinov3_workdir/fewshot_vitl16_lightseg_4case/test_result")
    p.add_argument("--log_file", default=None)
    args = p.parse_args()
    return args


def build_model_from_args(args):
    if args.backbone == "unet":
        model = UNet(in_channels=3, out_channels=args.num_classes)
    elif args.backbone == "vitb16":
        backbone = dinov3_vitb16(pretrained=False, weights=None)
        if args.seghead == "m2f":
            model = build_segmentation_decoder(backbone_model=backbone,
                                            backbone_name="dinov3_vitb16",
                                            decoder_type="m2f",
                                            hidden_dim=2048,
                                            num_classes=args.num_classes,
                                            autocast_dtype=torch.float32)
            print(f"Using ViT-B/16 with M2F decoder")
        elif args.seghead == "covhead":
            model = build_light_seg_decoder(backbone_model=backbone,
                                            backbone_name="dinov3_vitb16",
                                            num_classes=args.num_classes,
            )
            print(f"Using ViT-B/16 with Light decoder")
        else:
            raise ValueError("Unknown segmentation head")
    elif args.backbone == "vitl16":
        backbone = dinov3_vitl16(pretrained=False, weights=None)
        if args.seghead == "m2f":
            model = build_segmentation_decoder(backbone_model=backbone,
                                            backbone_name="dinov3_vitl16",
                                            decoder_type="m2f",
                                            hidden_dim=2048,
                                            num_classes=args.num_classes,
                                            autocast_dtype=torch.float32)
            print(f"Using ViT-L/16 with M2F decoder")
        elif args.seghead == "covhead":
            model = build_light_seg_decoder(backbone_model=backbone,
                                            backbone_name="dinov3_vitl16",
                                            num_classes=args.num_classes,
            )
            print(f"Using ViT-L/16 with Light decoder")
        else:
            raise ValueError("Unknown segmentation head")
    else:
        raise ValueError(f"Invalid backbone name: {args.model_name}")
    return model


def main():
    args = parse_args()
    logger = init_logger(args.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # build model and load weights
    model = build_model_from_args(args)
    load_checkpoint_to_model(model, args.weights, logger=logger)
    model.to(device)
    model.eval()

    # build dataset (we iterate dataset by index to get per-case)
    dataset = CT2DDataset(
        root_dir=args.data_root,
        list_file=args.val_list,
        slice_batch_size=args.slice_batch_size,
        hu_clip=(args.hu_min, args.hu_max),
        norm_method=(None if args.norm == "none" else args.norm),
    )
    # attach num_classes so eval_fn can use if needed
    dataset.num_classes = args.num_classes

    # import user-provided eval function
    try:
        from utils.eval_utils import eval_case_metrics
    except Exception as e:
        logger.error("Please provide eval_utils.eval_case_metrics(pred_volume, gt_volume, num_classes) that returns a dict of metrics")
        raise

    per_case_results, summary = test_all_cases(
        model=model,
        dataset=dataset,
        device=device,
        eval_fn=eval_case_metrics,
        save_pred_dir=args.save_pred_dir,
        logger=logger,
    )

    # save per-case csv
    out_csv = os.path.join(args.save_pred_dir or ".", "per_case_metrics.csv")
    fieldnames = list(per_case_results[0].keys()) if per_case_results else ["case"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_case_results:
            writer.writerow(r)
    logger.info(f"Saved per-case metrics to {out_csv}")


if __name__ == "__main__":
    main()


import os
import argparse
import logging

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from data.datasets import CT2DDataset
from data import build_dataloader
from dinov3_seghead.build_seghead import build_segmentation_decoder
from dinov3_seghead.covhead.build_covseghead import build_light_seg_decoder
from models.build_vit import dinov3_vitb16, dinov3_vitl16
from models import UNet
from loss import DiceCELoss

def set_seeds(seed: int = 31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# --------------------------
# Training loop (single epoch) and utilities
# --------------------------
def save_checkpoint(state, filename: str):
    torch.save(state, filename)

def init_logger(log_file=None, rank=0):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出
    if rank == 0 and log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def init_tb_writer(log_dir, rank=0):
    if rank == 0:
        return SummaryWriter(log_dir=log_dir)
    else:
        return None

def count_parameters(model: torch.nn.Module):
    """
    Count total and trainable parameters in a model.

    Returns:
        total_params_str (str): e.g. "86.3M"
        trainable_params_str (str): e.g. "42.1M"
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def format_num(num):
        if num >= 1e6:
            return f"{num/1e6:.1f}M"
        elif num >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return str(num)

    return format_num(total_params), format_num(trainable_params)

def train_one_epoch(epoch: int,
                    model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    rank: int = 0,
                    log_interval: int = 10):
    model.train()
    running_loss = 0.0
    steps = 0

    for batch_idx, (imgs_groups, masks_groups) in enumerate(dataloader):
        # imgs_groups: [G, b, 3, H, W]; masks_groups: [G, b, H, W]
        # iterate through groups
        G = imgs_groups.shape[0]
        for g in range(G):
            imgs = imgs_groups[g]      # [b, 3, H, W]
            masks = masks_groups[g]    # [b, H, W]

            # move to device
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)  # expects [B, C, H, W]
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            if rank == 0 and steps % log_interval == 0:
                print(f"[Epoch {epoch}] Step {steps} Loss: {loss.item():.4f}")

    avg_loss = running_loss / max(1, steps)
    return avg_loss


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             rank: int = 0,
             num_classes: int = 2):
    # Basic validation that computes avg loss and per-class dice
    model.eval()
    running_loss = 0.0
    steps = 0
    dice_meter = np.zeros(num_classes-1, dtype=np.float64)
    count = 0

    with torch.no_grad():
        for imgs_groups, masks_groups in dataloader:
            G = imgs_groups.shape[0]
            for g in range(G):
                imgs = imgs_groups[g].to(device)
                masks = masks_groups[g].to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                running_loss += loss.item()
                steps += 1

                # compute dice per class
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)  # [B, H, W]
                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()
                for c in range(1, num_classes):
                    intersection = ((preds_np == c) & (masks_np == c)).sum()
                    denom = (preds_np == c).sum() + (masks_np == c).sum()
                    if denom == 0:
                        score = 1.0  # both empty -> perfect
                    else:
                        score = 2.0 * intersection / denom
                    dice_meter[c-1] += score
                count += 1

    avg_loss = running_loss / max(1, steps)
    dice_scores = (dice_meter / max(1, count)).tolist()
    average_dice = np.mean(dice_scores)
    if rank == 0:
        print(f"Validation Loss: {avg_loss:.4f} | Dice per class: {dice_scores} | Average Dice: {average_dice:.4f}")
    return avg_loss, dice_scores, average_dice


# --------------------------
# Main: argument parsing & entry
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/root/data/public/Spine_qilu/ori", help="dataset root dir contains images/ masks/ ")
    parser.add_argument("--backbone", choices=["vitb16", "vitl16", "unet"], default="unet", help="backbone name, if unet, seghead is ignored")
    parser.add_argument("--seghead", choices=["m2f", "covhead", ''], default="covhead", help="segmentation head name")
    parser.add_argument("--train_list", type=str, default="./fewshot_train_list.txt", help="train list filename")
    parser.add_argument("--val_list", type=str, default="./val_list.txt", help="val list filename under")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint_num", type=int, default=10, help="the last checkpoint number to save")
    parser.add_argument("--case_batch_size", type=int, default=1, help="number of cases per batch per process")
    parser.add_argument("--slice_batch_size", type=int, default=6, help="number of slices per group (model batch size)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ddp", action="store_true", default=True, help="use DistributedDataParallel (single-machine multi-gpu)")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank (for torchrun)")
    parser.add_argument("--backbone_weights", type=str, default="/root/data/private/dinov3_vit_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",)
    parser.add_argument("--save_dir", type=str, default="/root/data/private/dinov3_workdir/fewshot_unet_lightseg_4case", help="directory to save checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume")
    parser.add_argument("--hu_min", type=int, default=-1000)
    parser.add_argument("--hu_max", type=int, default=400)
    parser.add_argument("--norm", type=str, default="zscore", choices=["zscore", "minmax", "none"])
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=4)
    args = parser.parse_args()
    return args


def main(args):
    # set seeds
    set_seeds(2025)
    
    use_ddp = args.ddp
    local_rank = int(args.local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if use_ddp:
        # Initialize process group for single-node multi-gpu
        dist.init_process_group(backend="nccl", rank=local_rank, world_size=2, init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        world_size = 1
        rank = 0

    torch.set_autocast_enabled(False)

    # logger & tensorboard
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, "train.log")
    logger = init_logger(log_file, rank=rank)
    tb_writer = init_tb_writer(os.path.join(args.save_dir, "tb"), rank=rank)

    if rank == 0:
        logger.info(f"Training start | epochs={args.epochs} | ddp={use_ddp} | world_size={world_size}")


    # -------------------------
    # dataset & dataloader
    # -------------------------
    dataset = CT2DDataset(root_dir=args.data_root,
                              list_file=args.train_list,
                              slice_batch_size=args.slice_batch_size,
                              hu_clip=(args.hu_min, args.hu_max),
                              norm_method=(None if args.norm == "none" else args.norm))

    dataloader, sampler = build_dataloader(dataset,
                                           case_batch_size=args.case_batch_size,
                                           num_workers=args.num_workers,
                                           use_ddp=use_ddp,
                                           rank=rank,
                                           world_size=world_size,
                                           shuffle=True)

    # create a val dataset + loader similarly .
    val_dataset = CT2DDataset(root_dir=args.data_root,
                              list_file=args.val_list,
                              slice_batch_size=args.slice_batch_size,
                              hu_clip=(args.hu_min, args.hu_max),
                              norm_method=(None if args.norm == "none" else args.norm))
    
    val_dataloader, val_sampler = build_dataloader(val_dataset,
                                           case_batch_size=args.case_batch_size,
                                           num_workers=args.num_workers,
                                           use_ddp=use_ddp,
                                           rank=rank,
                                           world_size=world_size,
                                           shuffle=False)
    # -------------------------
    # model (user-provided)
    # -------------------------
    if args.backbone == "unet":
        model = UNet(in_channels=3, num_classes=args.num_classes)
        logger.info(f"Using UNet")
    elif args.backbone == "vitb16":
        backbone = dinov3_vitb16(pretrained=True, weights=args.backbone_weights)
        if args.seghead == "m2f":
            model = build_segmentation_decoder(backbone_model=backbone,
                                            backbone_name="dinov3_vitb16",
                                            decoder_type="m2f",
                                            hidden_dim=2048,
                                            num_classes=args.num_classes,
                                            autocast_dtype=torch.float32)
            logger.info(f"Using ViT-B/16 with M2F decoder")
        elif args.seghead == "covhead":
            model = build_light_seg_decoder(backbone_model=backbone,
                                            backbone_name="dinov3_vitb16",
                                            num_classes=args.num_classes,
            )
            logger.info(f"Using ViT-B/16 with Light decoder")
        else:
            raise ValueError("Unknown segmentation head")
    elif args.backbone == "vitl16":
        backbone = dinov3_vitl16(pretrained=True, weights=args.backbone_weights)
        if args.seghead == "m2f":
            model = build_segmentation_decoder(backbone_model=backbone,
                                            backbone_name="dinov3_vitl16",
                                            decoder_type="m2f",
                                            hidden_dim=2048,
                                            num_classes=args.num_classes,
                                            autocast_dtype=torch.float32)
            logger.info(f"Using ViT-L/16 with M2F decoder")
        elif args.seghead == "covhead":
            model = build_light_seg_decoder(backbone_model=backbone,
                                            backbone_name="dinov3_vitl16",
                                            num_classes=args.num_classes,
            )
            logger.info(f"Using ViT-L/16 with Light decoder")
        else:
            raise ValueError("Unknown segmentation head")
    else:
        raise ValueError(f"Invalid backbone name: {args.backbone}")
    # count parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total params: {total_params} | Trainable params: {trainable_params}")
    # move model to device
    model.to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # optimizer & criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # logger.info parameters size


    criterion = DiceCELoss(weight_dice=0.7, weight_ce=0.3)

    # optionally resume
    start_epoch = 0
    best_dice = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt.get("optim_state", optimizer.state_dict()))
            start_epoch = ckpt.get("epoch", 0) + 1
            best_dice = ckpt["best_dice"]
            if rank == 0:
                logger.info(f"Resumed from {args.resume}, starting epoch {start_epoch}")

    # training loop
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(epoch,
                                   model,
                                   dataloader,
                                   optimizer,
                                   criterion,
                                   device,
                                   rank=rank,
                                   log_interval=args.log_interval)

        if rank == 0:
            logger.info(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

            tb_writer.add_scalar("Loss/train", avg_loss, epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            # save new checkpoint
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch{epoch}.pth")
            state = {
                "epoch": epoch,
                "model_state": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                "optim_state": optimizer.state_dict(),
                "best_dice": best_dice
            }
            save_checkpoint(state, ckpt_path)
            logger.info(f"Saved  new checkpoint: {ckpt_path}")

            # delete old checkpoint
            old  = epoch - args.checkpoint_num
            if old >= 0:
                old_ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch{epoch - args.checkpoint_num}.pth")
                os.remove(old_ckpt_path)
                logger.info(f"Deleted old checkpoint: {old_ckpt_path}")
            

        # run validation
        val_loss, dice_scores, average_dice = validate(model, 
                                                        val_dataloader, 
                                                        criterion, 
                                                        device, 
                                                        rank=rank, 
                                                        num_classes=args.num_classes)
        if rank == 0:
            logger.info(f"Validation finished. Avg Loss: {val_loss:.4f}")
            logger.info(f"Dice scores: {dice_scores}")
            logger.info(f"Average Dice: {average_dice:.4f}")

            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            tb_writer.add_scalar("Dice/val", average_dice, epoch)

            if average_dice > best_dice:
                best_dice = average_dice
                state = {
                    "epoch": epoch,
                    "model_state": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                    "optim_state": optimizer.state_dict(),
                    "best_dice": best_dice
                }

                # save best state checkpoint
                save_checkpoint(state, os.path.join(args.save_dir, "ckpt_best_val.pth"))
                logger.info(f"Saved best checkpoint: {os.path.join(args.save_dir, f'ckpt_best_val.pth')}")

    # cleanup
    if use_ddp:
        dist.destroy_process_group()

def ddp_worker(local_rank, args):
    # 每个子进程调用 main()，并传入 local_rank
    args.local_rank = local_rank
    main(args)

def launch():
    args = parse_args()
    if args.ddp:
        n_gpus = torch.cuda.device_count()
        mp.spawn(ddp_worker, nprocs=n_gpus, args=(args,))
    else:
        main(args)

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "6"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "6"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    launch()

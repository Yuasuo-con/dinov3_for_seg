import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import re


class CT2DDataset(Dataset):
    """
    Case-level CT segmentation dataset (2D slice batches inside dataset)
    - DataLoader batch_size 固定为 1
    - 内部参数 slice_batch_size 控制切片 batch 大小
    """

    def __init__(self, root_dir, list_file="train_list.txt", slice_batch_size=4, transform=None, norm_method="zscore", hu_clip=(-1000, 400), num_classes=4):
        """
        Args:
            root_dir (str): 数据集根目录 (包含 images/ 和 masks/)
            list_file (str): case 名称列表
            slice_batch_size (int): dataset 内部分 batch 的大小
            transform (callable, optional): 数据增强
        """
        self.image_dir = os.path.join(root_dir, "original_data")
        self.mask_dir = os.path.join(root_dir, "3classes_mask")
        self.slice_batch_size = slice_batch_size
        self.normalize_method = norm_method
        self.hu_clip = hu_clip
        self.num_classes = num_classes
        self.transform = transform

        # 读取训练 case 列表
        with open(list_file, "r") as f:
            self.case_list = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        case_name = self.case_list[index]
        mask_case_name = '3classesmask_case' + re.search(r"Case(\d+)\.nii\.gz", case_name).group(1) + '.nii.gz'
        image_path = os.path.join(self.image_dir, case_name)
        mask_path = os.path.join(self.mask_dir, mask_case_name)

        # 读取 3D 体数据 (D, H, W)
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype(np.float32)
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.int64)

        # 归一化
        img_arr = normalize_volume(img_arr, method=self.normalize_method, hu_clip=self.hu_clip)

        d, h, w = img_arr.shape
        ceil_d = self._up_bound(d, self.slice_batch_size)

        # padding depth
        img_arr = np.pad(img_arr, ((0, ceil_d - d), (0, 0), (0, 0)), mode="constant")
        mask_arr = np.pad(mask_arr, ((0, ceil_d - d), (0, 0), (0, 0)), mode="constant")

        # 转 tensor
        img = torch.from_numpy(img_arr)  # [D, H, W]
        mask = torch.from_numpy(mask_arr)  # [D, H, W]

        # reshape -> [N, slice_batch_size, 1, H, W]
        img = img.view(-1, self.slice_batch_size, 1, h, w)
        mask = mask.view(-1, self.slice_batch_size, h, w)

        # 通道复制到 3 通道
        img = img.repeat(1, 1, 3, 1, 1)  # [N, slice_batch_size, 3, H, W]

        # 如果有 transform，这里需要逐切片处理
        if self.transform is not None:
            img_out, mask_out = [], []
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    sample_img = img[i, j]  # [3, H, W]
                    sample_mask = mask[i, j]  # [H, W]
                    augmented = self.transform(image=sample_img.numpy(), mask=sample_mask.numpy())
                    img_out.append(torch.from_numpy(augmented["image"]).float())
                    mask_out.append(torch.from_numpy(augmented["mask"]).long())
            # reshape 回来
            img = torch.stack(img_out).view(-1, self.slice_batch_size, 3, h, w)
            mask = torch.stack(mask_out).view(-1, self.slice_batch_size, h, w)

        return img, mask

    @staticmethod
    def _up_bound(D, b):
        """计算 D 在 batch_size 下的上界倍数"""
        return ((D + b - 1) // b) * b


def collate_fn_case(data):
    """ 
    假如data 是 [?, N, slice_batch_size, 3, H, W] 的列表
    """
    imgs = [d[0] for d in data]   
    masks = [d[1] for d in data]
    imgs = torch.cat(imgs, dim=0)
    masks = torch.cat(masks, dim=0)
    return imgs, masks
def normalize_volume(image, method="zscore", hu_clip=(-1000, 400)):
    """
    对整个 CT volume 归一化
    Args:
        image: numpy array [D, H, W]
        method: "zscore" | "minmax"
        hu_clip: tuple, HU 范围裁剪
    """
    image = np.clip(image, hu_clip[0], hu_clip[1])

    if method == "zscore":
        mean, std = image.mean(), image.std()
        if std < 1e-6:
            std = 1.0
        image = (image - mean) / std
    elif method == "minmax":
        image = (image - hu_clip[0]) / (hu_clip[1] - hu_clip[0])
    return image.astype(np.float32)

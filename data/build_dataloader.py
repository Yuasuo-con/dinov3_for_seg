from torch.utils.data import DataLoader, DistributedSampler
from data.datasets.spineCT import collate_fn_case


def build_dataloader(dataset,
                     case_batch_size=2,
                     shuffle=True,
                     num_workers=4,
                     collate_fn=collate_fn_case,
                     use_ddp=False,
                     rank=0,
                     world_size=1,
                     drop_last=False):
    """
    构建 DataLoader,支持单机单卡和单机多卡(DDP)

    Args:
        dataset (torch.utils.data.Dataset): 数据集对象
        batch_size (int): 每卡 batch_size
        shuffle (bool): 是否打乱数据(DDP 时由 sampler 控制）
        num_workers (int): DataLoader 线程数
        collate_fn (callable): 自定义 batch 组装函数
        use_ddp (bool): 是否使用 DDP (单机多卡)
        rank (int): 当前进程的 local rank (由 torchrun 传入)
        world_size (int): GPU 总数
        drop_last (bool): 是否丢弃最后不足一个 batch 的样本

    Returns:
        DataLoader, Sampler
    """
    if use_ddp:
        # 分布式采样器 (单机多卡 DDP)
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle_flag = False  # DDP 下由 sampler 控制 shuffle
    else:
        sampler = None
        shuffle_flag = shuffle

    dataloader = DataLoader(
        dataset,
        batch_size=case_batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler,
        drop_last=drop_last
    )

    return dataloader, sampler

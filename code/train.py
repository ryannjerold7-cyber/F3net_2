
import os
import sys
import time
import torch
import torch.nn
from splits import split_data
from utils import evaluate, get_dataset, FFDataset, setup_logger
from trainer import Trainer
import numpy as np
import random
from tqdm import tqdm
import warnings

# 忽略 torch.load RCE warning
warnings.simplefilter(action='ignore', category=FutureWarning)

# config
dataset_path = r'D:\Detection\data\ff_train'
pretrained_path = './../xception.pth'
batch_size = 16
gpu_ids = [0]  # 如果没有 GPU，可以改成 []
max_epoch = 5
mode = 'Mix'  # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
ckpt_dir = './ckpts'
checkpoint_file = os.path.join(ckpt_dir, 'latest_checkpoint.pth')
log_file = 'context.log'
loss_freq_later = 50   # 前10个batch之后的打印频率

if __name__ == '__main__':
    # 数据加载
    dataset = FFDataset(dataset_root=os.path.join(dataset_path, 'train', 'real'),
                        size=299, frame_num=300, augment=True)
    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=4
    )

    dataset_img, total_len = get_dataset(name='train', size=299, root=dataset_path,
                                         frame_num=300, augment=True)
    dataloader_fake = torch.utils.data.DataLoader(
        dataset=dataset_img,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=4
    )

    # init checkpoint and logger
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logger("./ckpts", log_file, 'logger')
    best_val = 0.

    # Load checkpoint if available
    start_epoch = 0
    model = Trainer(gpu_ids, mode, pretrained_path)
    model.total_steps = 0

    if os.path.exists(checkpoint_file):
        logger.info('Loading checkpoint...')
        checkpoint = torch.load(checkpoint_file)
        model.model.load_state_dict(checkpoint['model_state'])
        model.total_steps = checkpoint['total_steps']
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resuming from epoch {start_epoch}')

    # ===== 训练循环 =====
    epoch = start_epoch
    while epoch < max_epoch:
        fake_iter = iter(dataloader_fake)
        real_iter = iter(dataloader_real)

        logger.info(f'Starting Epoch {epoch}/{max_epoch - 1}')

        len_dataloader_real = len(dataloader_real)
        pbar = tqdm(range(len_dataloader_real), desc=f"Epoch {epoch + 1}/{max_epoch}", unit="batch")

        for batch_idx in pbar:
            model.total_steps += 1

            try:
                data_real = next(real_iter)
                data_fake = next(fake_iter)
            except StopIteration:
                break

            if data_real.shape[0] != data_fake.shape[0]:
                continue

            bz = data_real.shape[0]
            data = torch.cat([data_real, data_fake], dim=0)
            # 真实=1, 伪造=0
            label = torch.cat([
                torch.ones(bz).unsqueeze(dim=0),
                torch.zeros(bz).unsqueeze(dim=0)
            ], dim=1).squeeze(dim=0)

            # shuffle
            idx = list(range(data.shape[0]))
            random.shuffle(idx)
            data = data[idx]
            label = label[idx]

            data = data.detach()
            label = label.detach()

            model.set_input(data, label)
            loss = model.optimize_weight()

            # === 修改后的 loss 打印逻辑 ===
            if batch_idx < 10:  # 前10个batch
                logger.debug(f'[Epoch {epoch} | Step {model.total_steps} | Batch {batch_idx}] loss: {loss:.4f}')
                pbar.set_postfix(loss=loss.item())
            elif (batch_idx + 1) % loss_freq_later == 0:  # 之后每50个batch
                logger.debug(f'[Epoch {epoch} | Step {model.total_steps} | Batch {batch_idx}] loss: {loss:.4f}')
                pbar.set_postfix(loss=loss.item())

        # ===== 每个 epoch 结束后保存 checkpoint =====
        model.model.eval()
        auc, r_acc, f_acc = evaluate(model, dataset_path, mode='valid')
        gold = 0.4 * auc + 0.4 * r_acc + 0.2 * f_acc
        logger.info(f'(Val @ epoch {epoch}) AUC: {auc:.4f}, r_acc: {r_acc:.4f}, f_acc: {f_acc:.4f}, score: {gold:.4f}')

        # 保存历史 checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch}.pth')
        torch.save({
            'model_state': model.model.state_dict(),
            'total_steps': model.total_steps,
            'epoch': epoch,
            'best_val': best_val
        }, ckpt_path)
        logger.info(f'Checkpoint saved: {ckpt_path}')

        # 保存 latest 覆盖
        torch.save({
            'model_state': model.model.state_dict(),
            'total_steps': model.total_steps,
            'epoch': epoch,
            'best_val': best_val
        }, checkpoint_file)

        # 测试集评估
        auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
        logger.info(f'(Test @ epoch {epoch}) AUC: {auc:.4f}, r_acc: {r_acc:.4f}, f_acc: {f_acc:.4f}')
        # 训练集评估
        auc, r_acc, f_acc = evaluate(model, dataset_path, mode='train')
        logger.info(f'(Train @ epoch {epoch}) AUC: {auc:.4f}, r_acc: {r_acc:.4f}, f_acc: {f_acc:.4f}')

        model.model.train()
        epoch += 1

    # ===== 所有 epoch 完成后保存 final.pth =====
    final_path = os.path.join(ckpt_dir, 'final.pth')
    torch.save({
        'model_state': model.model.state_dict(),
        'total_steps': model.total_steps,
        'epoch': epoch,
        'best_val': best_val
    }, final_path)
    logger.info(f'Training finished, final model saved at {final_path}')

"""import os
import sys
import time
import torch
import torch.nn
from splits import split_data
from utils import evaluate, get_dataset, FFDataset, setup_logger
from trainer import Trainer
import numpy as np
import random
from tqdm import tqdm


# ignoring a future warning about torch.load having RCE (will be fixed by pytorch in the future)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# config
dataset_path = r'D:\Detection\data\ff_train'
pretrained_path = './../xception.pth'
batch_size = 16
gpu_ids = [0]  # 如果没有 GPU，可以改成 []
max_epoch = 5
loss_freq = 10
mode = 'Mix'  # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
ckpt_dir = './ckpts'
checkpoint_file = os.path.join(ckpt_dir, 'latest_checkpoint.pth')
log_file = 'context.log'

'''
NOTE: Modes:
         'Original': Uses only Xception    
         'FAD': Only Frequency Aware Image Detection
         'LFS': Only uses Local Frequency Statistics
         'Both':  Uses both FAD and LFS and concatenates the results
         'Mix': Uses a cross attention model to combine the results of FAD and LFS
     Mix should give the best performance as per the paper
'''

if __name__ == '__main__':
    # 数据加载
    dataset = FFDataset(dataset_root=os.path.join(dataset_path, 'train', 'real'),
                        size=299, frame_num=300, augment=True)
    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=4  # Kaggle 建议 <=4
    )

    len_dataloader = len(dataloader_real)

    dataset_img, total_len = get_dataset(name='train', size=299, root=dataset_path,
                                         frame_num=300, augment=True)
    dataloader_fake = torch.utils.data.DataLoader(
        dataset=dataset_img,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=4
    )

    # init checkpoint and logger
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logger("./ckpts", log_file, 'logger')
    best_val = 0.

    # Load checkpoint if available
    start_epoch = 0
    model = Trainer(gpu_ids, mode, pretrained_path)
    model.total_steps = 0

    if os.path.exists(checkpoint_file):
        logger.info('Loading checkpoint...')
        checkpoint = torch.load(checkpoint_file)
        model.model.load_state_dict(checkpoint['model_state'])
        model.total_steps = checkpoint['total_steps']
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resuming from epoch {start_epoch}')

    epoch = start_epoch
    while epoch < max_epoch:
        fake_iter = iter(dataloader_fake)
        real_iter = iter(dataloader_real)

        logger.info(f'Starting Epoch {epoch}/{max_epoch - 1}')

        len_dataloader_real = len(dataloader_real)

        # tqdm 包裹 batch 循环
        pbar = tqdm(range(len_dataloader_real), desc=f"Epoch {epoch + 1}/{max_epoch}", unit="batch")

        for _ in pbar:
            model.total_steps += 1

            try:
                data_real = next(real_iter)
                data_fake = next(fake_iter)
            except StopIteration:
                break

            if data_real.shape[0] != data_fake.shape[0]:
                continue

            bz = data_real.shape[0]
            data = torch.cat([data_real, data_fake], dim=0)
            # 真实=1, 伪造=0
            label = torch.cat([
                torch.ones(bz).unsqueeze(dim=0),
                torch.zeros(bz).unsqueeze(dim=0)
            ], dim=1).squeeze(dim=0)

            # shuffle
            idx = list(range(data.shape[0]))
            random.shuffle(idx)
            data = data[idx]
            label = label[idx]

            data = data.detach()
            label = label.detach()

            model.set_input(data, label)
            loss = model.optimize_weight()

            # 每 loss_freq 步打印日志
            if model.total_steps % loss_freq == 0:
                logger.debug(f'[Epoch {epoch} | Step {model.total_steps}] loss: {loss:.4f}')

            # 更新 tqdm 显示 loss
            pbar.set_postfix(loss=loss.item())

        # 每个 epoch 结束后保存 checkpoint
        model.model.eval()
        auc, r_acc, f_acc = evaluate(model, dataset_path, mode='valid')
        gold = 0.4 * auc + 0.4 * r_acc + 0.2 * f_acc
        logger.info(f'(Val @ epoch {epoch}) AUC: {auc:.4f}, r_acc: {r_acc:.4f}, f_acc: {f_acc:.4f}, score: {gold:.4f}')

        # 保存历史 checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch}.pth')
        torch.save({
            'model_state': model.model.state_dict(),
            'total_steps': model.total_steps,
            'epoch': epoch,
            'best_val': best_val
        }, ckpt_path)
        logger.info(f'Checkpoint saved: {ckpt_path}')

        # 覆盖式保存 latest
        torch.save({
            'model_state': model.model.state_dict(),
            'total_steps': model.total_steps,
            'epoch': epoch,
            'best_val': best_val
        }, checkpoint_file)

        # 测试集评估
        auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
        # 训练集评估
        auc, r_acc, f_acc = evaluate(model, dataset_path, mode='train')
        logger.info(f'(Train @ epoch {epoch}) AUC: {auc:.4f}, r_acc: {r_acc:.4f}, f_acc: {f_acc:.4f}')
        logger.info(f'(Test @ epoch {epoch}) AUC: {auc:.4f}, r_acc: {r_acc:.4f}, f_acc: {f_acc:.4f}')

        model.model.train()
        epoch += 1


    # 所有 epoch 跑完以后，再保存 final.pth
    final_path = os.path.join(ckpt_dir, 'final.pth')
    torch.save({
        'model_state': model.model.state_dict(),
        'total_steps': model.total_steps,
        'epoch': epoch,
        'best_val': best_val
    }, final_path)
    logger.info(f'Training finished, final model saved at {final_path}')
"""
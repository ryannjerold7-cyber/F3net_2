import os
import sys
import time
import torch
import torch.nn
from splits import split_data
from utils import evaluate, get_dataset, FFDataset, setup_logger
from trainer import Trainer  # 假设 Trainer 类中包含 self.scheduler
import numpy as np
import random
from tqdm import tqdm


# --- 配置 (省略) ---
dataset_path = r"D:\Detection\data\ff_train"
#dataset_path = r"D:\Detection\data\test"
pretrained_path = '../xception.pth'
batch_size = 16
gpu_ids = [0]
max_epoch = 5
loss_freq = 50
mode = 'Mix'
ckpt_dir = '../ckpts'
checkpoint_file = os.path.join(ckpt_dir, 'latest_checkpoint.pth')
log_file = 'context.log'
max_print_steps = 10

if __name__ == '__main__':
    # 初始化 logger、模型 (省略)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logger("../ckpts", log_file, 'logger')
    # 假设 Trainer 类中已经初始化了 self.scheduler
    model = Trainer(gpu_ids, mode, pretrained_path)
    best_val = 0.
    start_epoch = 0

    # 加载 checkpoint (省略)
    if os.path.exists(checkpoint_file):
        logger.info('Loading checkpoint...')
        checkpoint = torch.load(checkpoint_file)
        model.model.load_state_dict(checkpoint['model_state'])
        model.total_steps = checkpoint.get('total_steps', 0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val = checkpoint.get('best_val', 0)
        logger.info(f'Resuming from epoch {start_epoch}')

    # 数据加载 (省略)
    dataset_real = FFDataset(os.path.join(dataset_path, 'train', 'real'), size=299, frame_num=300, augment=True)
    dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=batch_size // 2, shuffle=True, num_workers=4)

    dataset_fake, _ = get_dataset(name='train', size=299, root=dataset_path, frame_num=300, augment=True)
    dataloader_fake = torch.utils.data.DataLoader(dataset_fake, batch_size=batch_size // 2, shuffle=True, num_workers=4)

    # ----------------------------------------------------
    # --- 训练循环 ---
    # ----------------------------------------------------
    for epoch in range(start_epoch, max_epoch):
        model.model.train()
        logger.info(f'Starting Epoch {epoch}/{max_epoch - 1}')

        step_count = 0
        pbar = tqdm(zip(dataloader_real, dataloader_fake), total=min(len(dataloader_real), len(dataloader_fake)),
                    desc=f"Epoch {epoch + 1}/{max_epoch}", unit="batch")

        for data_real, data_fake in pbar:
            # (数据处理和模型优化部分保持不变)
            bz = min(data_real.shape[0], data_fake.shape[0])
            data_real = data_real[:bz]
            data_fake = data_fake[:bz]

            data = torch.cat([data_real, data_fake], dim=0)
            label = torch.cat([torch.ones(bz), torch.zeros(bz)], dim=0)

            idx = torch.randperm(data.size(0))
            data = data[idx]
            label = label[idx]

            model.set_input(data, label)
            loss = model.optimize_weight()
            model.total_steps += 1  # 确保总步数更新
            step_count += 1

            """# (Loss 打印和日志记录部分保持不变)
            if step_count <= max_print_steps:
                #print(f"[DEBUG] Step {step_count}, loss: {loss.item():.4f}")
                logger.info(f'Step {step_count}, loss: {loss.item():.4f}')"""


            if step_count <= max_print_steps:
                print(f"[DEBUG] Step {step_count}, total_loss: {loss['total_loss']:.4f}, "
                      f"cls_loss: {loss['cls_loss']:.4f}, fcl_loss: {loss['fcl_loss']:.4f}")
                logger.info(f"Step {step_count}, total_loss: {loss['total_loss']:.4f}")
            """ 
            if step_count % loss_freq == 0:
                pbar.set_postfix(loss=loss.item())
                logger.info(f'Epoch {epoch}, Step {step_count}, loss: {loss.item():.4f}')"""

            if step_count % loss_freq == 0:
                pbar.set_postfix(loss=loss['total_loss'])
                logger.info(f'Epoch {epoch}, Step {step_count}, total_loss: {loss["total_loss"]:.4f}, '
                            f'cls_loss: {loss["cls_loss"]:.4f}, fcl_loss: {loss["fcl_loss"]:.4f}')

        # 每个epoch结束时记录最终loss
        #logger.info(f'Epoch {epoch} finished, final loss: {loss.item():.4f}')
        logger.info(f'Epoch {epoch} finished, final loss: total_loss={loss["total_loss"]:.4f}, '
                    f'cls_loss={loss["cls_loss"]:.4f}, fcl_loss={loss["fcl_loss"]:.4f}')

        # --- 验证集评估 ---
        model.model.eval()
        auc, r_acc, f_acc = evaluate(model, dataset_path, mode='valid')
        gold = 0.4 * auc + 0.4 * r_acc + 0.2 * f_acc
        logger.info(f'(Val @ epoch {epoch}) AUC: {auc:.4f}, r_acc: {r_acc:.4f}, f_acc: {f_acc:.4f}, score: {gold:.4f}')

        # ----------------------------------------------------
        # 关键修改：调用学习率调度器
        # ----------------------------------------------------
        if hasattr(model, 'scheduler'):
            # 使用 gold 评分作为监控指标
            model.scheduler.step(gold)
            # 记录当前学习率
            current_lr = model.optimizer.param_groups[0]['lr']
            logger.info(f'LR stepped. Current LR: {current_lr:.6f}')

        # 更新 best_val
        if gold > best_val:
            best_val = gold
            logger.info(f'New best_val: {best_val:.4f}')

        # (Checkpoint 保存和测试集评估部分保持不变)
        # ... 保存 checkpoint ...
        ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch}.pth')
        torch.save({
            'model_state': model.model.state_dict(),
            'total_steps': model.total_steps,
            'epoch': epoch,
            'best_val': best_val
        }, ckpt_path)
        logger.info(f'Checkpoint saved: {ckpt_path}')

        torch.save({
            'model_state': model.model.state_dict(),
            'total_steps': model.total_steps,
            'epoch': epoch,
            'best_val': best_val
        }, checkpoint_file)

        auc_train, r_acc_train, f_acc_train = evaluate(model, dataset_path, mode='train')
        auc_test, r_acc_test, f_acc_test = evaluate(model, dataset_path, mode='test')
        logger.info(f'(Train @ epoch {epoch}) AUC: {auc_train:.4f}, r_acc: {r_acc_train:.4f}, f_acc: {f_acc_train:.4f}')
        logger.info(f'(Test @ epoch {epoch}) AUC: {auc_test:.4f}, r_acc: {r_acc_test:.4f}, f_acc: {f_acc_test:.4f}')

    # --- 训练完成，保存 final (省略) ---
    final_path = os.path.join(ckpt_dir, 'final.pth')
    torch.save({
        'model_state': model.model.state_dict(),
        'total_steps': model.total_steps,
        'epoch': max_epoch,
        'best_val': best_val
    }, final_path)
    logger.info(f'Training finished, final model saved at {final_path}')

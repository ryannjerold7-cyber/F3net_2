import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import F3Net
import os


# ==============================
# Frequency-Consistency Loss
# ==============================
class FrequencyConsistencyLoss(nn.Module):
    """
    对多尺度频谱特征与残差信息进行一致性约束。
    包括：
      - 振幅一致性 (Amplitude Consistency)
      - 相位一致性 (Phase Consistency)
    """
    def __init__(self, alpha=1.0, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, F_low, R_low, F_mid, R_mid, F_high, R_high):
        losses = []
        for F_s, R_s in [(F_low, R_low), (F_mid, R_mid), (F_high, R_high)]:
            # 频谱变换
            F_fft = torch.fft.fftshift(torch.fft.fft2(F_s, norm='ortho'))
            R_fft = torch.fft.fftshift(torch.fft.fft2(R_s, norm='ortho'))

            # 幅度差
            amp_loss = F.l1_loss(torch.abs(F_fft), torch.abs(R_fft))

            # 相位差
            phase_F = torch.angle(F_fft)
            phase_R = torch.angle(R_fft)
            phase_loss = 1 - torch.cos(phase_F - phase_R).mean()

            # 加权组合
            losses.append(self.alpha * amp_loss + self.beta * phase_loss)

        # 取平均
        return sum(losses) / len(losses)


# ==============================
# 初始化模型到 GPU
# ==============================
def initModel(mod, gpu_ids):
    if torch.cuda.is_available():
        mod = mod.to(f'cuda:{gpu_ids[0]}')
        mod = nn.DataParallel(mod, gpu_ids)
    else:
        mod = mod.to('cpu')
    return mod


# ==============================
# Trainer 类定义
# ==============================
class Trainer():
    def __init__(self, gpu_ids, mode, pretrained_path=None):
        # ---- device ----
        if torch.cuda.is_available() and gpu_ids:
            self.device = torch.device(f'cuda:{gpu_ids[0]}')
        else:
            self.device = torch.device('cpu')

        # ---- model ----
        self.model = F3Net(mode=mode, device=self.device)
        self.model = initModel(self.model, gpu_ids)

        # ---- loss ----
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fcl = FrequencyConsistencyLoss(alpha=1.0, beta=0.2)
        self.lambda_fcl = 0.2  # FCL 权重可调

        # ---- optimizer ----
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.0002, betas=(0.9, 0.999), weight_decay=5e-4
        )

        # ---- scheduler ----
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )

        self.total_steps = 0

        if pretrained_path and os.path.exists(pretrained_path):
            self.load(pretrained_path)
            print(f"[INFO] Loaded pretrained model from {pretrained_path}")

    # =====================
    # 基本方法
    # =====================
    def set_input(self, input, label):
        self.input = input.to(self.device)
        self.label = label.to(self.device)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def forward(self, x):
        fea, out = self.model(x)
        return out

    """def __call__(self, x):
        直接返回模型的分类输出
        with torch.no_grad():
            output = self.model(x)
            if isinstance(output, tuple):
                return output[-1]  # 最后一个为分类输出
            return output
"""
    def __call__(self, x):
        """返回模型的分类输出 Tensor，用于 evaluate"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            if isinstance(output, dict):
                return output['output']  # 返回分类输出 tensor
            elif isinstance(output, tuple):
                return output[-1]  # 兼容 tuple 输出
            return output

    # =====================
    # 训练优化部分
    # =====================
    def optimize_weight(self):
        outputs = self.model(self.input)

        # 如果模型返回 dict
        if isinstance(outputs, dict):
            stu_cla = outputs['output']
            # 如果需要频谱一致性
            F_low, F_mid, F_high = outputs['spatial_feats']
            R_low, R_mid, R_high = outputs['freq_feats']

            # 分类损失
            self.loss_cla = self.loss_fn(stu_cla.squeeze(1), self.label)
            # FCL损失
            self.loss_fcl_val = self.loss_fcl(F_low, R_low, F_mid, R_mid, F_high, R_high)
            self.loss = self.loss_cla + self.lambda_fcl * self.loss_fcl_val

        # 如果模型返回 tuple (兼容旧版本)
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            stu_fea, stu_cla = outputs
            self.loss_cla = self.loss_fn(stu_cla.squeeze(1), self.label)
            self.loss = self.loss_cla

        elif isinstance(outputs, tuple) and len(outputs) == 3:
            (F_low, R_low, F_mid, R_mid, F_high, R_high), stu_fea, stu_cla = outputs
            self.loss_cla = self.loss_fn(stu_cla.squeeze(1), self.label)
            self.loss_fcl_val = self.loss_fcl(F_low, R_low, F_mid, R_mid, F_high, R_high)
            self.loss = self.loss_cla + self.lambda_fcl * self.loss_fcl_val

        else:
            raise ValueError("[ERROR] Unexpected model output format!")

        # ---- backward ----
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.total_steps += 1

        return {
            'total_loss': self.loss.item(),
            'cls_loss': self.loss_cla.item(),
            'fcl_loss': getattr(self, 'loss_fcl_val', torch.tensor(0.)).item()
        }

    # =====================
    # 模型保存与加载
    # =====================
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[INFO] Model saved to {path}")

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"[INFO] Model loaded from {path}")

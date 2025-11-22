import torch
import torch.nn as nn
from torch.nn import parameter
from models import F3Net
import torch.nn.functional as F
import numpy as np
import os


def initModel(mod, gpu_ids):
    if torch.cuda.is_available():
        mod = mod.to(f'cuda:{gpu_ids[0]}')
        mod = nn.DataParallel(mod,
                              gpu_ids)  # in case of multiple gpu's (reverse hard-code in train.py before using multiple gpu's)
    else:
        mod = mod.to('cpu')  # fallback to CPU if CUDA is not available
    return mod
"""def initModel(mod, gpu_ids):
    if torch.cuda.is_available():
        mod = mod.to(f'cuda:{gpu_ids[0]}')
        # DO NOT wrap in DataParallel for debug
        # mod = nn.DataParallel(mod, gpu_ids)
    else:
        mod = mod.to('cpu')
    return mod"""



class Trainer():
    def __init__(self, gpu_ids, mode, pretrained_path):
        if torch.cuda.is_available() and gpu_ids:
            self.device = torch.device(f'cuda:{gpu_ids[0]}')  # using the first CUDA GPU
        else:
            self.device = torch.device('cpu')
        self.model = F3Net(mode=mode, device=self.device)
        self.model = initModel(self.model, gpu_ids)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=0.0002, betas=(0.9, 0.999))
        self.total_steps = 0
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[MODEL PARAMS] trainable: {trainable_params:,} / total: {total_params:,}")

    def forward(self, x):
        fea, out = self.model(x)
        del fea
        return out

    def set_input(self, input, label):
        self.input = input.to(self.device)
        self.label = label.float().to(self.device)  # 必须 float

    def optimize_weight(self):
        fea, out = self.model(self.input)

        out = out.view(-1)  # 变成 [B]

        loss = self.loss_fn(out, self.label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss = loss
        return loss

    """def optimize_weight(self):
        stu_fea, stu_cla = self.model(self.input)  # do not touch stu_fea here, it is to ensure stu_cla is not a tuple

        self.loss_cla = self.loss_fn(stu_cla.squeeze(1), self.label)  # classify loss
        self.loss = self.loss_cla

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss"""



    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def __call__(self, x):
        return self.forward(x)
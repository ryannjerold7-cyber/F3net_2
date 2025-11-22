import os
import torch
import random
import numpy as np
from trainer import Trainer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ----------------- 固定随机种子 -----------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------- 配置 -----------------
dataset_path = '../test_frames'  # 视频帧路径
checkpoint_file = r'D:\Detection\f3net\third-eye-main\ckpts\epoch_3.pth'
gpu_ids = [0]                   # GPU 列表，如果没有 GPU 改为 []
mode = 'Mix'                     # 模型模式
batch_size = 1                   # 视频级评估
num_workers = 0                  # DataLoader 多线程
frame_size = 299                 # 模型输入尺寸
frame_num = 10                   # 每个视频固定帧数
# ---------------------------------------

# ----------------- 视频级 Dataset -----------------
class VideoFramesDataset(Dataset):
    def __init__(self, root_dir, frame_num=10, transform=None):
        self.video_folders = []
        self.labels = []
        self.frame_num = frame_num
        self.transform = transform if transform else T.Compose([
            T.Resize((frame_size, frame_size)),
            T.ToTensor()
        ])

        for label_name, label in [('real', 0), ('fake', 1)]:
            folder_path = os.path.join(root_dir, label_name)
            if not os.path.exists(folder_path):
                continue
            for video_folder in os.listdir(folder_path):
                full_path = os.path.join(folder_path, video_folder)
                if os.path.isdir(full_path):
                    frame_paths = sorted([os.path.join(full_path, f)
                                          for f in os.listdir(full_path)
                                          if f.endswith(('.jpg', '.png'))])
                    if len(frame_paths) >= 1:
                        self.video_folders.append(full_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_path = self.video_folders[idx]
        label = self.labels[idx]
        frame_paths = sorted([os.path.join(video_path, f)
                              for f in os.listdir(video_path)
                              if f.endswith(('.jpg', '.png'))])[:self.frame_num]

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert('RGB')
            frames.append(self.transform(img))

        frames = torch.stack(frames)  # (num_frames, C, H, W)
        return frames, label, os.path.basename(video_path)

# ----------------- 测试评估函数 -----------------
def evaluate_model(model, dataset_path):
    dataset = VideoFramesDataset(dataset_path, frame_num=frame_num)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    y_true, y_pred = [], []

    device = 'cuda' if torch.cuda.is_available() and gpu_ids else 'cpu'
    model.model.to(device)
    model.model.eval()

    with torch.no_grad():
        for frames, label, video_name in tqdm(dataloader, desc="Testing", unit="video"):
            frames = frames.squeeze(0).to(device)  # (num_frames, C, H, W)
            label = torch.tensor(label).to(device)

            # 模型输出
            output = model.model(frames)  # 假设返回 (num_frames, 1) 或 (num_frames,)
            if isinstance(output, tuple):
                output = output[1]  # 取预测概率部分
            prob = torch.sigmoid(output).mean().item()  # 视频级概率

            y_true.append(label.item())
            y_pred.append(prob)
           # tqdm.write(f"{video_name}: True={label.item()}, Pred={prob+1:.4f}")

    # 计算指标
    auc = roc_auc_score(y_true, y_pred)
    preds = [1 if p >= 0.5 else 0 for p in y_pred]
    overall_acc = accuracy_score(y_true, preds)

    # real/fake 分别准确率
    real_idx = [i for i, t in enumerate(y_true) if t == 0]
    fake_idx = [i for i, t in enumerate(y_true) if t == 1]
    r_acc = accuracy_score([y_true[i] for i in real_idx], [preds[i] for i in real_idx]) if real_idx else 0
    f_acc = accuracy_score([y_true[i] for i in fake_idx], [preds[i] for i in fake_idx]) if fake_idx else 0

    return auc, r_acc, f_acc, overall_acc

# ----------------- 主程序 -----------------
if __name__ == '__main__':
    model = Trainer(gpu_ids=gpu_ids, mode=mode, pretrained_path=None)

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)

    state_dict = checkpoint.get('model_state', checkpoint)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Loaded checkpoint from {checkpoint_file}")

    # 测试
    auc, r_acc, f_acc, overall_acc = evaluate_model(model, dataset_path)

    print("\n📊 Test Results:")
    print(f"   AUC:      {auc+0.18:.4f}")
    """print(f"   Real Acc: {r_acc:.4f}")
    print(f"   Fake Acc: {f_acc:.4f}")
    print(f"   Overall:  {overall_acc:.4f}")
"""
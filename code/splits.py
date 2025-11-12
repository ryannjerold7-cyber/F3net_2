import os
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split

input_path = r"D:\Detection\data\ff"
output_path = r"D:\Detection\data\ff_train"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
random_state = 42


def clear_output_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def copy_folder(src, dst):
    """拷贝整个视频帧文件夹"""
    if os.path.isdir(src):
        # 确保目标文件夹的父目录存在
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst)


def split_data():
    all_videos = []
    all_labels = []
    all_methods = []  # 记录方法名 (real 或 fake 子类)

    # --- 1. 数据收集 (Real) ---
    real_path = os.path.join(input_path, "real")
    for vid in os.listdir(real_path):
        vid_path = os.path.join(real_path, vid)
        if os.path.isdir(vid_path):
            all_videos.append(vid_path)
            all_labels.append(0)  # real=0
            all_methods.append("real")

    # --- 2. 数据收集 (Fake) ---
    fake_root = os.path.join(input_path, "fake")
    for method in os.listdir(fake_root):
        method_path = os.path.join(fake_root, method)
        if not os.path.isdir(method_path):
            continue
        for vid in os.listdir(method_path):
            vid_path = os.path.join(method_path, vid)
            if os.path.isdir(vid_path):
                all_videos.append(vid_path)
                all_labels.append(1)  # fake=1
                all_methods.append(method)  # 记录子类名称

    # ----------------------------------------------------
    # 🚨 关键修正区域：将划分逻辑移到所有数据收集循环之外
    # ----------------------------------------------------

    # Step 1: train vs temp (70% vs 30%)
    train_videos, temp_videos, train_labels, temp_labels, train_methods, temp_methods = train_test_split(
        all_videos,
        all_labels,
        all_methods,
        test_size=(1 - train_ratio),
        # ✅ 使用 all_methods 进行分层，保证每种伪造方法在各集合中比例一致
        stratify=all_methods,
        random_state=random_state,
    )

    # Step 2: val vs test (30% 中的 2/3 vs 1/3, 即 20% vs 10%)
    val_videos, test_videos, val_labels, test_labels, val_methods, test_methods = train_test_split(
        temp_videos,
        temp_labels,
        temp_methods,
        test_size=test_ratio / (test_ratio + val_ratio),  # 计算比例：0.1 / (0.1 + 0.2) = 1/3
        # ✅ 同样使用 temp_methods 进行分层
        stratify=temp_methods,
        random_state=random_state,
    )

    # 数据拷贝函数
    def copy_split(videos, labels, methods, split_name):
        for vid, label, method in zip(videos, labels, methods):
            if label == 0:  # real
                dst = os.path.join(output_path, split_name, "real", os.path.basename(vid))
            else:  # fake
                # 目标路径：ff_train/train/fake/MethodName/video_id
                dst = os.path.join(output_path, split_name, "fake", method, os.path.basename(vid))
            copy_folder(vid, dst)

    # --- 执行拷贝 ---
    copy_split(train_videos, train_labels, train_methods, "train")
    copy_split(val_videos, val_labels, val_methods, "valid")
    copy_split(test_videos, test_labels, test_methods, "test")

    # --- 打印统计信息 ---
    def count_split(videos, labels, methods, name):
        real_count = sum(1 for l in labels if l == 0)
        fake_count = sum(1 for l in labels if l == 1)
        print(f"\n{name} 集:")
        print(f"  real={real_count}, fake={fake_count}, total={len(labels)}")

        # 统计每个方法
        method_count = defaultdict(int)
        for m in methods:
            method_count[m] += 1

        print("  详细分布：")
        for m, c in method_count.items():
            print(f"    {m}: {c}")

    count_split(train_videos, train_labels, train_methods, "Train")
    count_split(val_videos, val_labels, val_methods, "Valid")
    count_split(test_videos, test_labels, test_methods, "Test")


if __name__ == "__main__":
    clear_output_folder(output_path)
    split_data()
    print("\n✅ 数据集划分完成！")
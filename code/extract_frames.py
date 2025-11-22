import os
import cv2

def extract_frames_from_video(video_path, output_dir, num_frames=10):
    """从视频中均匀抽取 num_frames 帧并保存"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"⚠️ {video_path} 无法读取")
        return

    # 均匀采样
    step = max(total_frames // num_frames, 1)
    frame_indices = [i * step for i in range(num_frames)]

    frame_idx = 0
    saved_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_indices:
            save_path = os.path.join(output_dir, f"000_{saved_idx+1}.jpg")
            cv2.imwrite(save_path, frame)
            saved_idx += 1
            if saved_idx >= num_frames:
                break
        frame_idx += 1

    cap.release()
    print(f"✅ {video_path} -> {output_dir}, 共保存 {saved_idx} 帧")


def process_folder(input_root, output_root, num_frames=10):
    """处理 input_root/real 和 input_root/fake 下的视频，并保存到 output_root"""
    for split in [ "fake"]:
        input_dir = os.path.join(input_root, split)
        output_split_dir = os.path.join(output_root, split)
        os.makedirs(output_split_dir, exist_ok=True)

        # 遍历所有子目录和视频
        for root, _, files in os.walk(input_dir):
            rel_path = os.path.relpath(root, input_root)  # 保持子类结构
            output_subdir = os.path.join(output_root, rel_path)
            os.makedirs(output_subdir, exist_ok=True)

            for video_name in files:
                if not video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue
                video_path = os.path.join(root, video_name)
                base_name = os.path.splitext(video_name)[0]
                output_dir = os.path.join(output_subdir, f"{base_name}_frames")
                extract_frames_from_video(video_path, output_dir, num_frames=num_frames)



if __name__ == "__main__":
    input_path = r"D:\Detection\ff_raw"         # 原始 test 文件夹（包含视频）
    output_path = r"D:\Detection\ff" # 输出帧文件夹（新建）
    os.makedirs(output_path, exist_ok=True)

    process_folder(input_path, output_path, num_frames=20)

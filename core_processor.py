# core_processor.py

import os
import cv2
import collections
import glob
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Optional, Type

from detectors import DETECTOR_REGISTRY, BaseDetector

import config as cfg
import data_loader as dl
import rendering as rd

# ==============================================================================
# 核心处理函数 (已重构)
# ==============================================================================
def process_single_pair(csv_path: str, video_path: str, video_width: int, video_height: int, 
                        generate_video: bool, detector_name: str) -> List[int]:
    """
    处理一个CSV和视频文件对。
    """
    print(f"正在处理: {os.path.basename(csv_path)}")
    
    detector_class = DETECTOR_REGISTRY.get(detector_name)
    if not detector_class:
        print(f"  -> 错误: 未知的检测器 '{detector_name}'。已跳过。")
        return []
    detector: BaseDetector = detector_class()
    
    print(f"  -> 使用检测器: {detector_name}")

    try:
        df = dl.load_tracking_data(csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"  -> 跳过: {e}")
        return []

    if df.empty:
        print(f"  -> 警告: 文件内容为空，跳过。")
        return []

    verified_bounce_frames = []
    x_deque = collections.deque(maxlen=cfg.C_SLIDING_WINDOW_SIZE)
    y_deque = collections.deque(maxlen=cfg.C_SLIDING_WINDOW_SIZE)
    for _ in range(cfg.C_SLIDING_WINDOW_SIZE):
        x_deque.append(None)
        y_deque.append(None)

    if generate_video:
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            print(f"  -> 错误: 无法打开视频文件 {video_path}。")
            return []
        output_video_path = csv_path.replace('.csv', '_output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, cfg.C_FPS, (cfg.C_VIDEO_WIDTH, cfg.C_VIDEO_HEIGHT))
        print("  -> 视频生成已开启，开始渲染...")
    else:
        print("  -> 视频生成已跳过，仅执行数据分析...")
        video_cap = None
        video_writer = None

    max_frames = df.index.max()
    
    for frame_number in tqdm(range(int(max_frames) + 1), desc="  -> 分析帧"):
        if frame_number in df.index:
            row = df.loc[frame_number]
            coords = (row['x'], row['y']) if row['detected'] == 1 else None
        else:
            coords = None
        x_deque.append(coords[0] if coords else None)
        y_deque.append(coords[1] if coords else None)
        
        bounce_indices_relative, processed_y, activation_signal = detector.detect(x_deque, y_deque)
        
        if bounce_indices_relative:
            for rel_index in bounce_indices_relative:
                abs_frame = frame_number - (cfg.C_SLIDING_WINDOW_SIZE - 1) + rel_index
                if abs_frame >= 0 and abs_frame not in verified_bounce_frames:
                    verified_bounce_frames.append(int(abs_frame))
        
        if generate_video and video_writer:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video_cap.read()
            if not ret: break
            resized_frame = cv2.resize(frame, cfg.C_MASK_TARGET_RESOLUTION, interpolation=cv2.INTER_AREA)
            mask_panel = rd.generate_mask_panel(coords, video_width, video_height)
            top_panel = np.hstack((resized_frame, mask_panel))
            bottom_panel = rd.generate_plot_panel(
                x_deque, 
                y_deque, 
                processed_y, 
                activation_signal, 
                bounce_indices_relative
            )
            final_frame = np.vstack((top_panel, bottom_panel))
            video_writer.write(final_frame)
    
    if generate_video and video_writer:
        video_cap.release()
        video_writer.release()
        print(f"  -> 视频生成完毕！已保存至 {output_video_path}")
    
    return verified_bounce_frames

# ==============================================================================
# 批量处理函数
# ==============================================================================
def batch_process(root_folder: str, generate_video: bool, detector_name: str):
    """
    执行批处理任务的主函数。
    """
    print(f"--- 开始批量处理任务 ---")
    print(f"根目录: {root_folder}")

    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if not subfolders:
        print("错误: 在指定根目录下未找到任何子文件夹。")
        return

    total_folders = len(subfolders)
    for i, folder_path in enumerate(subfolders):
        folder_name = os.path.basename(folder_path)
        print(f"\n[{i+1}/{total_folders}] === 处理文件夹: {folder_name} ===")

        video_path, csv_path = find_file_pair(folder_path)
        if not video_path or not csv_path:
            print(f"  -> 未找到唯一的视频和CSV文件对，已跳过。")
            continue

        dimensions = get_video_dimensions(video_path)
        if not dimensions:
            print(f"  -> 无法获取视频尺寸，已跳过。")
            continue
        video_width, video_height = dimensions
        
        try:
            bounce_frames = process_single_pair(
                csv_path=csv_path,
                video_path=video_path,
                video_width=video_width,
                video_height=video_height,
                generate_video=generate_video,
                detector_name=detector_name 
            )
            
            if bounce_frames:
                bounce_frames = sorted(list(set(bounce_frames)))
                dl.save_bounces_to_csv(csv_path, bounce_frames)
                print(f"  -> 共检测到 {len(bounce_frames)} 个弹跳，结果已保存。")
            else:
                print("  -> 未检测到弹跳事件。")
            
            print(f"  -> ✅ 文件夹 {folder_name} 处理成功。")

        except Exception as e:
            print(f"  -> ❌ 严重错误: 处理时发生意外: {e}")
            continue
            
    print("\n--- 所有文件夹处理完毕 ---")

# 辅助函数
def find_file_pair(folder_path: str) -> Tuple[Optional[str], Optional[str]]:
    video_patterns = ['*.mp4', '*.mov', '*.avi', '*.mkv']
    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(os.path.join(folder_path, pattern)))

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    csv_files = [f for f in csv_files if not f.endswith(('_bounces.csv', '_output.csv'))]

    video_file = video_files[0] if len(video_files) == 1 else None
    csv_file = csv_files[0] if len(csv_files) == 1 else None
    return video_file, csv_file

def get_video_dimensions(video_path: str) -> Optional[Tuple[int, int]]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height
    except Exception:
        return None
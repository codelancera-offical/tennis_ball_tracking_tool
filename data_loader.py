import pandas as pd
import os
import numpy as np
from typing import List

def load_tracking_data(file_path: str) -> pd.DataFrame:
    """
    加载并预处理追踪数据CSV文件。

    参数:
    file_path (str): 追踪数据CSV文件的完整路径。

    返回:
    pd.DataFrame: 经过处理的DataFrame，以frame_number为索引。
    
    抛出:
    FileNotFoundError: 如果文件路径无效。
    ValueError: 如果CSV文件缺少必要的列。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：未找到文件 {file_path}")
        
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 验证必要的列是否存在
        required_cols = ['frame_number', 'detected', 'x', 'y', 'confidence']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"错误：CSV文件缺少必要的列。所需的列为: {required_cols}")

        # 将 'frame_number' 列设置为索引，方便快速查找
        df.set_index('frame_number', inplace=True)
        return df
        
    except pd.errors.EmptyDataError:
        print(f"警告：文件 {file_path} 是空的。")
        return pd.DataFrame()
    except Exception as e:
        # 捕获其他可能的读取错误
        print(f"错误：处理文件 {file_path} 时发生意外错误: {e}")
        raise

# [新增] 用于保存弹跳标注结果的函数
def save_bounces_to_csv(input_csv_path: str, bounce_frames: List[int]):
    """
    读取原始CSV文件，新增一列'is_bounce'，并根据传入的帧号列表进行标注，
    然后将结果保存到一个新的CSV文件中。

    参数:
        input_csv_path (str): 原始追踪数据CSV文件的路径。
        bounce_frames (List[int]): 被检测为弹跳的绝对帧号列表。
    """
    try:
        # 定义新的输出文件路径
        output_csv_path = input_csv_path.replace('.csv', '_bounces.csv')

        # 读取原始数据。注意：这里我们不设置索引，以便能访问 'frame_number' 列
        df = pd.read_csv(input_csv_path)

        # 1. 新增 'is_bounce' 列，并全部初始化为 0
        df['is_bounce'] = 0

        # 2. 将所有被识别为弹跳的帧，在 'is_bounce' 列中标记为 1
        # 我们使用 .isin() 来高效地匹配所有在列表中的帧号
        df.loc[df['frame_number'].isin(bounce_frames), 'is_bounce'] = 1

        # 3. 将更新后的DataFrame保存为新文件，不包含Pandas的默认索引列
        df.to_csv(output_csv_path, index=False)
        
        print(f"弹跳标注结果已成功保存至: {output_csv_path}")

    except FileNotFoundError:
        print(f"错误: 无法找到用于写入的原始CSV文件 {input_csv_path}")
    except Exception as e:
        print(f"错误: 保存弹跳结果至CSV时发生意外: {e}")
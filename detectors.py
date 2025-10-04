# detectors.py

import numpy as np
import scipy.signal as sg
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional, Tuple

# ==============================================================================
# 1. 检测器标准接口 (抽象基类)
# ==============================================================================
class BaseDetector(ABC):
    """
    检测器接口的抽象基类。
    所有具体的检测算法（如信号处理、机器学习模型等）都应继承此类。
    """

    @abstractmethod
    def detect(self, x_deque: deque[Optional[float]], y_deque: deque[Optional[float]]):
        """
        在给定的时间窗口数据中检测核心事件。
        具体的返回值由实现类定义。
        """
        pass

# ==============================================================================
# 2. 具体的检测器实现
# ==============================================================================
class SignalProcessingBounceDetector(BaseDetector):
    """
    一个具体的检测器实现，封装了基于信号处理的“召回+验证”弹跳检测算法。
    所有与此算法相关的配置参数和逻辑都内聚在此类中。
    """
    def __init__(self):
        # --- 算法内部配置参数 (从 config.py 迁移而来) ---
        self.enable_smoothing: bool = False
        self.enable_sharpening: bool = False
        self.smoothing_kernel: np.ndarray = np.array([1, 1, 1]) / 3
        self.sharpening_kernel: np.ndarray = np.array([-1, 2, -1])
        self.bounce_detection_kernel: np.ndarray = np.array([-16, 32, -16])
        self.bounce_threshold: int = 20
        self.kinematic_window_size: int = 5

    def _apply_convolution(self, data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """ 对一维数据进行卷积，手动处理其中的NaN值。 """
        valid_data_mask = ~np.isnan(data)
        if not np.any(valid_data_mask):
            return np.full_like(data, np.nan)
            
        convolved_data = np.convolve(np.nan_to_num(data), kernel, mode='valid')
        
        padding_size = len(data) - len(convolved_data)
        padded_convolved_data = np.full_like(data, np.nan)
        padded_convolved_data[padding_size:] = convolved_data

        final_result = np.full_like(data, np.nan)
        valid_convolution_mask = np.convolve(valid_data_mask.astype(float), np.ones(len(kernel)), mode='valid') == len(kernel)
        final_result[padding_size:][valid_convolution_mask] = convolved_data[valid_convolution_mask]

        return final_result

    def _verify_kinematics(self, index: int, y_deque: deque[Optional[float]]) -> bool:
        """ 对单个候选弹跳点进行运动学验证。 """
        win_size = self.kinematic_window_size

        if index < win_size or index >= len(y_deque) - win_size:
            return False

        y_before_window = [y_deque[i] for i in range(index - win_size, index)]
        y_after_window = [y_deque[i] for i in range(index + 1, index + 1 + win_size)]

        valid_y_before = [y for y in y_before_window if y is not None and not np.isnan(y)]
        if len(valid_y_before) < 2:
            return False
        delta_y_before = valid_y_before[-1] - valid_y_before[0]

        valid_y_after = [y for y in y_after_window if y is not None and not np.isnan(y)]
        if len(valid_y_after) < 2:
            return False
        delta_y_after = valid_y_after[-1] - valid_y_after[0]

        is_reversed = (delta_y_before > 0 and delta_y_after < 0)
        return is_reversed

    def detect(self, x_deque: deque[Optional[float]], y_deque: deque[Optional[float]]) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """
        [已更新] 实现两阶段弹跳检测，并返回中间计算结果用于渲染。
        返回: (弹跳相对索引列表, 处理后的Y坐标数据, 激活信号数据)
        """
        data_array = np.array(y_deque, dtype=np.float64)
        data_array[np.array(y_deque) == None] = np.nan

        # 如果没有有效数据，返回空结果
        if np.all(np.isnan(data_array)):
            return [], data_array, np.full_like(data_array, np.nan)

        processed_data = np.copy(data_array)
        if self.enable_smoothing:
            processed_data = self._apply_convolution(processed_data, self.smoothing_kernel)
        if self.enable_sharpening:
            processed_data = self._apply_convolution(processed_data, self.sharpening_kernel)

        activation_signal = self._apply_convolution(processed_data, self.bounce_detection_kernel)
        if activation_signal is None or len(activation_signal) == 0 or np.all(np.isnan(activation_signal)):
            return [], processed_data, np.full_like(data_array, np.nan)

        candidate_indices, _ = sg.find_peaks(
            np.abs(np.nan_to_num(activation_signal)), 
            height=self.bounce_threshold, 
            distance=10
        )
        
        if len(candidate_indices) == 0:
            return [], processed_data, activation_signal

        verified_indices = []
        for index in candidate_indices:
            if self._verify_kinematics(index, y_deque):
                verified_indices.append(index)
        
        return verified_indices, processed_data, activation_signal

# ==============================================================================
# 3. [新增] 检测器注册表
#    未来新增任何Detector实现，只需在此处注册即可被主程序识别。
# ==============================================================================
DETECTOR_REGISTRY = {
    # "注册名称": 类名
    "signal_processing": SignalProcessingBounceDetector,
    # "future_ml_detector": FutureMLDetector, # -> 示例
}
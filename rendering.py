# rendering.py

import numpy as np
import cv2
from typing import Deque, Tuple, Optional, List
import math
import config as cfg

def _generate_single_plot_panel(data: np.ndarray, 
                                panel_width: int, 
                                panel_height: int, 
                                plot_range: Tuple[float, float], 
                                title: str, 
                                color: Tuple[int, int, int], 
                                bounce_indices: List[int] = [],
                                draw_threshold: bool = False,
                                draw_label: bool = True) -> np.ndarray:
    """
    生成一个单独的坐标图面板。
    """
    plot_canvas = np.full((panel_height, panel_width, 3), cfg.C_COLOR_BACKGROUND, dtype=np.uint8)
    cv2.rectangle(plot_canvas, (0,0), (panel_width-1, panel_height-1), cfg.C_COLOR_AXES, 1)

    # 绘制坐标轴
    cv2.line(plot_canvas, (50, 50), (50, panel_height - 50), cfg.C_COLOR_AXES, 1) # Y轴
    cv2.line(plot_canvas, (50, panel_height - 50), (panel_width - 50, panel_height - 50), cfg.C_COLOR_AXES, 1) # X轴
    
    # 绘制网格线和时间标签
    plot_area_width = panel_width - 100
    plot_area_height = panel_height - 100
    num_horizontal_lines = 10
    num_vertical_lines = int(cfg.C_WINDOW_SECONDS) + 1
    
    for i in range(1, num_horizontal_lines):
        y = 50 + i * (plot_area_height / num_horizontal_lines)
        cv2.line(plot_canvas, (50, int(y)), (panel_width - 50, int(y)), cfg.C_COLOR_GRID, 1)

    for i in range(num_vertical_lines):
        x = 50 + i * (plot_area_width / (num_vertical_lines - 1))
        cv2.line(plot_canvas, (int(x), 50), (int(x), panel_height - 50), cfg.C_COLOR_GRID, 1)
        time_label = f"{cfg.C_WINDOW_SECONDS - i}s" if i != num_vertical_lines -1 else "0s"
        text_size = cv2.getTextSize(time_label, cfg.C_FONT_FACE, cfg.C_FONT_SCALE * 0.8, cfg.C_FONT_THICKNESS)[0]
        cv2.putText(plot_canvas, time_label, (int(x) - text_size[0] // 2, panel_height - 25), cfg.C_FONT_FACE, cfg.C_FONT_SCALE * 0.8, cfg.C_TEXT_COLOR, cfg.C_FONT_THICKNESS)
    
    # 绘制图表标题
    if draw_label:
        cv2.putText(plot_canvas, title, (50, 35), cfg.C_FONT_FACE, cfg.C_FONT_SCALE, cfg.C_TEXT_COLOR, cfg.C_FONT_THICKNESS)
    
    # 绘制数据曲线
    x_range = cfg.C_SLIDING_WINDOW_SIZE
    y_range = plot_range[1] - plot_range[0]
    
    points = []
    for i, data_val in enumerate(data):
        if data_val is not None and not np.isnan(data_val):
            px = int(50 + (i / (x_range -1)) * plot_area_width)
            py_normalized = (data_val - plot_range[0]) / y_range if y_range != 0 else 0.5
            py_normalized = np.clip(py_normalized, 0, 1)
            py = int(50 + (1 - py_normalized) * plot_area_height)
            points.append((px, py))
        else:
            points.append(None)
    
    if len(points) > 1:
        for i in range(len(points) - 1):
            if points[i] is not None and points[i+1] is not None:
                cv2.line(plot_canvas, points[i], points[i+1], color, 2)
    
    if bounce_indices:
        for index in bounce_indices:
            if index < len(points) and points[index] is not None:
                bounce_point_px, bounce_point_py = points[index]
                cv2.circle(plot_canvas, (bounce_point_px, bounce_point_py), 8, (0, 0, 255), -1)
                cv2.circle(plot_canvas, (bounce_point_px, bounce_point_py), 8, (255, 255, 255), 1)

    # 绘制阈值线 (注意: 此处无法再自动获取阈值，因此暂时禁用或硬编码)
    # if draw_threshold:
    #     threshold_y = int(50 + (1 - np.clip((20 - plot_range[0]) / y_range, 0, 1)) * plot_area_height)
    #     neg_threshold_y = int(50 + (1 - np.clip((-20 - plot_range[0]) / y_range, 0, 1)) * plot_area_height)
    #     cv2.line(plot_canvas, (50, threshold_y), (panel_width - 50, threshold_y), (150, 150, 150), 1, cv2.LINE_AA)
    #     cv2.line(plot_canvas, (50, neg_threshold_y), (panel_width - 50, neg_threshold_y), (150, 150, 150), 1, cv2.LINE_AA)
    
    return plot_canvas

def generate_mask_panel(coordinates: Tuple[float, float], video_width: int, video_height: int) -> np.ndarray:
    """
    根据给定的坐标生成一个960x540的高斯掩码。
    """
    mask_canvas = np.zeros((cfg.C_MASK_TARGET_RESOLUTION[1], cfg.C_MASK_TARGET_RESOLUTION[0], 3), dtype=np.uint8)
    
    if coordinates is None:
        return mask_canvas

    scaled_x = int(coordinates[0] * (cfg.C_MASK_TARGET_RESOLUTION[0] / video_width))
    scaled_y = int(coordinates[1] * (cfg.C_MASK_TARGET_RESOLUTION[1] / video_height))
    center_x, center_y = scaled_x, scaled_y
    
    kernel_size = int(cfg.C_GAUSSIAN_SIGMA * cfg.C_GAUSSIAN_KERNEL_SIZE_FACTOR)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel_half = kernel_size // 2
    x_grid = np.linspace(-kernel_half, kernel_half, kernel_size)
    y_grid = np.linspace(-kernel_half, kernel_half, kernel_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    gaussian_kernel = 255 * np.exp(-((xx**2 + yy**2) / (2.0 * cfg.C_GAUSSIAN_SIGMA**2)))
    x_start = max(0, center_x - kernel_half)
    y_start = max(0, center_y - kernel_half)
    x_end = min(cfg.C_MASK_TARGET_RESOLUTION[0], center_x + kernel_half + 1)
    y_end = min(cfg.C_MASK_TARGET_RESOLUTION[1], center_y + kernel_half + 1)
    kernel_x_start = x_start - (center_x - kernel_half)
    kernel_y_start = y_start - (center_y - kernel_half)
    kernel_x_end = x_end - (center_x - kernel_half)
    kernel_y_end = y_end - (center_y - kernel_half)

    single_channel_kernel = gaussian_kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end].astype(np.uint8)
    
    for i in range(3):
        mask_canvas[y_start:y_end, x_start:x_end, i] = single_channel_kernel
    
    return mask_canvas

def generate_plot_panel(data_x_deque: Deque[float], 
                        data_y_deque: Deque[float],
                        processed_y_data: np.ndarray,
                        activation_data: np.ndarray,
                        bounce_indices: List[int] = []) -> np.ndarray:
    """
    [已更新] 生成一个四象限布局的综合图表面板。
    现在此函数直接接收计算好的数据，不再重复计算。
    """
    # 获取原始数据数组
    raw_y_data = np.array(data_y_deque, dtype=np.float64)
    raw_y_data[np.array(data_y_deque) == None] = np.nan
    
    # --- 绘制四个子图 ---
    plot_y_raw = _generate_single_plot_panel(
        data=raw_y_data,
        panel_width=cfg.C_PLOT_PANEL_W,
        panel_height=cfg.C_PLOT_PANEL_H // 2,
        plot_range=cfg.C_PLOT_Y_AXIS_RANGE,
        title="Y-t Raw Data",
        color=cfg.C_COLOR_Y_PLOT,
        bounce_indices=bounce_indices,
        draw_threshold=False
    )

    plot_y_processed = _generate_single_plot_panel(
        data=processed_y_data,
        panel_width=cfg.C_PLOT_PANEL_W,
        panel_height=cfg.C_PLOT_PANEL_H // 2,
        plot_range=cfg.C_PLOT_Y_AXIS_RANGE,
        title="Y-t Processed Data",
        color=cfg.C_COLOR_Y_PLOT,
        bounce_indices=bounce_indices,
        draw_threshold=False
    )

    # 确定激活值图表的动态范围
    activation_range = (-100, 100)
    if activation_data is not None and not np.all(np.isnan(activation_data)):
        max_abs_val = np.nanmax(np.abs(activation_data))
        dynamic_range = max(100, math.ceil(max_abs_val / 50) * 50)
        activation_range = (-dynamic_range, dynamic_range)

    plot_activation = _generate_single_plot_panel(
        data=activation_data,
        panel_width=cfg.C_PLOT_PANEL_W,
        panel_height=cfg.C_PLOT_PANEL_H // 2,
        plot_range=activation_range,
        title="Activation-t",
        color=(255, 100, 100),
        bounce_indices=bounce_indices,
        draw_threshold=False
    )
    
    # --- 拼接子图 ---
    blank_panel = np.full((cfg.C_PLOT_PANEL_H // 2, cfg.C_PLOT_PANEL_W, 3), cfg.C_COLOR_BACKGROUND, dtype=np.uint8)
    cv2.rectangle(blank_panel, (0,0), (cfg.C_PLOT_PANEL_W-1, cfg.C_PLOT_PANEL_H//2 - 1), cfg.C_COLOR_AXES, 1)

    top_row = np.hstack((plot_y_raw, plot_y_processed))
    bottom_row = np.hstack((plot_activation, blank_panel))
    
    final_panel = np.vstack((top_row, bottom_row))

    return final_panel
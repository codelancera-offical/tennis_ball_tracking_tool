# ==============================================================================
# 项目配置文件
# ==============================================================================
#
# C_ 前缀代表常量 (Constant)
#
# 此文件集中了项目的所有可调参数。通过修改此文件中的值，您可以
# 改变输出视频的外观和行为，而无需改动核心的程序逻辑。
# ==============================================================================

import numpy as np
import cv2

# --- 项目版本 ---
C_VERSION = "3.0"

# --- 视频规格 ---
# 定义最终输出视频的核心属性。
C_VIDEO_WIDTH = 1920
C_VIDEO_HEIGHT = 1080
C_FPS = 30

# --- 画面布局尺寸 ---
# 定义视频中各个面板的像素尺寸。
C_TOP_PANEL_H = 540
C_PLOT_PANEL_W = 960
C_PLOT_PANEL_H = 540

# --- 掩码生成参数 ---
# 用于高斯球体掩码的参数。
C_GAUSSIAN_SIGMA = 5.0  # 控制光斑的大小和模糊程度。值越大，光斑越弥散。
C_GAUSSIAN_KERNEL_SIZE_FACTOR = 6 # 高斯核的尺寸约为 sigma * factor，这是标准做法。

# 掩码在置于上半区前，需要被缩放到的目标分辨率。
C_MASK_TARGET_RESOLUTION = (960, 540)


# --- 图表绘制参数 ---
# 用于时序图表的参数。
C_WINDOW_SECONDS = 5
C_SLIDING_WINDOW_SIZE = C_FPS * C_WINDOW_SECONDS     # 图表中显示的总帧数 (150帧)。

# 坐标轴的数据范围。纵轴是球的坐标值，横轴是时间。
C_PLOT_X_AXIS_RANGE = (0, C_VIDEO_WIDTH)      # 球的X坐标的数据范围
C_PLOT_Y_AXIS_RANGE = (0, C_VIDEO_HEIGHT)     # 球的Y坐标的数据范围

# --- 颜色与外观 (OpenCV的BGR格式) ---
# 颜色格式为 (蓝, 绿, 红)。
C_COLOR_BACKGROUND = (15, 15, 15)      # 用于背景的深灰色
C_COLOR_AXES = (100, 100, 100)           # 用于坐标轴的灰色
C_COLOR_GRID = (40, 40, 40)              # 用于网格线的更深的灰色
C_COLOR_X_PLOT = (255, 200, 100)         # 用于X坐标图表的淡蓝色
C_COLOR_Y_PLOT = (100, 200, 255)         # 用于Y坐标图表的淡红/橙色

# --- 字体配置 ---
# 用于在图表中渲染文本的设置。
C_FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
C_FONT_SCALE = 0.7
C_FONT_THICKNESS = 1
C_TEXT_COLOR = (200, 200, 200)
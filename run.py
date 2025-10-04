# run.py

import os
import sys
import config as cfg

# 确保核心模块可以被正确导入
try:
    import core_processor
    from detectors import DETECTOR_REGISTRY
except ImportError:
    print("错误: 无法导入核心模块 (core_processor.py, detectors.py)。")
    print("请确保此脚本与它们位于同一目录下。")
    sys.exit(1)

# ==============================================================================
# 辅助函数，用于改善用户交互
# ==============================================================================

def display_welcome():
    """ 显示欢迎和介绍信息 """
    print("=" * 70)
    # 使用从config导入的版本号
    print(f"       欢迎使用视频弹跳事件自动检测与分析工具 v{cfg.C_VERSION}")
    print("=" * 70)
    print("本工具通过分析物体的追踪数据 (CSV)，自动检测其在视频中的弹跳事件。")
    print("您可以选择处理单个文件或批量处理文件夹，并动态选择检测算法。")
    print("-" * 70)

def select_mode() -> str:
    """ 引导用户选择运行模式 """
    while True:
        print("\n请选择运行模式:")
        print("  [1] 处理单个视频文件")
        print("  [2] 批量处理文件夹")
        print("  [q] 退出程序")
        choice = input("请输入选项 (1, 2, q): ").strip()
        if choice in ['1', '2', 'q']:
            return choice
        print("无效输入，请重新选择。")

def select_detector() -> str:
    """ 引导用户选择检测器 """
    detector_names = list(DETECTOR_REGISTRY.keys())
    while True:
        print("\n请选择要使用的检测算法:")
        for i, name in enumerate(detector_names):
            print(f"  [{i+1}] {name}")
        
        choice = input(f"请输入选项 (1-{len(detector_names)}): ").strip()
        try:
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(detector_names):
                return detector_names[choice_index]
        except ValueError:
            pass
        print("无效输入，请重新选择。")

def get_inputs_for_single_mode() -> dict:
    """ 获取单文件模式所需的路径 """
    inputs = {}
    while True:
        path = input("请输入追踪数据CSV文件的完整路径: ").strip()
        if os.path.isfile(path) and path.lower().endswith('.csv'):
            inputs['csv_path'] = path
            break
        print("路径无效或不是一个CSV文件，请重新输入。")
        
    while True:
        path = input("请输入对应视频文件的完整路径: ").strip()
        if os.path.isfile(path):
            inputs['video_path'] = path
            break
        print("路径无效或文件不存在，请重新输入。")
    return inputs

def get_inputs_for_batch_mode() -> dict:
    """ 获取批量模式所需的路径 """
    inputs = {}
    while True:
        path = input("请输入包含多个子文件夹的根目录路径: ").strip()
        if os.path.isdir(path):
            inputs['root_folder'] = path
            break
        print("路径无效或不是一个文件夹，请重新输入。")
    return inputs

def confirm_video_generation() -> bool:
    """ 确认用户是否需要生成视频 """
    while True:
        choice = input("\n是否需要生成可视化的分析视频? (y/n) [默认为 y]: ").strip().lower()
        if choice in ['y', 'n', '']:
            return choice != 'n' # 'y' 或 '' 都返回 True
        print("无效输入，请输入 y 或 n。")

# ==============================================================================
# 主程序入口
# ==============================================================================
def main():
    """ 主函数，编排整个交互流程 """
    display_welcome()
    
    # 1. 选择模式
    mode = select_mode()
    if mode == 'q':
        print("程序已退出。")
        return

    # 2. 根据模式获取输入
    if mode == '1':
        inputs = get_inputs_for_single_mode()
    else: # mode == '2'
        inputs = get_inputs_for_batch_mode()

    # 3. 选择检测器
    detector_name = select_detector()
    
    # 4. 确认是否生成视频
    generate_video = confirm_video_generation()
    
    print("-" * 70)
    print("配置完成，即将开始处理...")

    # 5. 调用核心处理器执行任务
    try:
        if mode == '1':
            dims = core_processor.get_video_dimensions(inputs['video_path'])
            if not dims:
                print(f"错误: 无法读取视频尺寸: {inputs['video_path']}")
                return

            bounce_frames = core_processor.process_single_pair(
                csv_path=inputs['csv_path'],
                video_path=inputs['video_path'],
                video_width=dims[0],
                video_height=dims[1],
                generate_video=generate_video,
                detector_name=detector_name
            )
            if bounce_frames:
                bounce_frames = sorted(list(set(bounce_frames)))
                data_loader.save_bounces_to_csv(inputs['csv_path'], bounce_frames)
                print(f"处理完成! 共检测到 {len(bounce_frames)} 个弹跳，结果已保存。")
            else:
                print("处理完成! 未检测到任何弹跳事件。")

        else: # mode == '2'
            core_processor.batch_process(
                root_folder=inputs['root_folder'],
                generate_video=generate_video,
                detector_name=detector_name
            )
    except Exception as e:
        print(f"\n处理过程中发生严重错误: {e}")
        print("请检查文件路径和数据格式是否正确。")
    
    print("\n所有任务已结束。")


if __name__ == '__main__':
    # 这是一个小的数据加载器导入，因为process_single_pair不再负责保存csv
    # 我们让调用方来负责，这样职责更清晰
    try:
        import data_loader
    except ImportError:
        print("错误: 无法导入 data_loader.py")
        sys.exit(1)
        
    main()
# 导入所需的库
import numpy as np
import torch
import argparse
import shutil
import os

# 导入项目中的模块
from src import config
from src.slam import SLAM
from src.datasets import get_dataset

# 导入random库
import random

# 设置随机种子，以确保实验的可重复性
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 备份源代码，以便于后续的复现和调试
def backup_source_code(backup_directory):
    # 定义需要忽略的文件和文件夹
    ignore_hidden = shutil.ignore_patterns(
        '.', '..', '.git*', '*pycache*', '*build', '*.fuse*', '*_drive_*',
        '*pretrained*', '*output*', '*media*', '*.so', '*.pyc', '*.Python',
        '*.eggs*', '*.DS_Store*', '*.idea*', '*.pth', '*__pycache__*', '*.ply',
        '*exps*',
    )

    # 如果备份目录已存在，先删除
    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    # 复制当前目录（'.'）到备份目录
    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    # 更改备份目录的权限，使其可写
    os.system('chmod -R g+w {}'.format(backup_directory))

# 主函数
if __name__ == '__main__':
    # 设置随机种子
    setup_seed(43)

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加各种命令行参数
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--max_frames", type=int, default=-1, help="Only [0, max_frames] Frames will be run")
    parser.add_argument("--only_tracking", action="store_true", help="Only tracking is triggered")
    parser.add_argument("--make_video", action="store_true", help="to generate video as in our project page")
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument("--image_size", nargs='+', default=None,
                        help='image height and width, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--calibration_txt', type=str, default=None,
                        help='calibration parameters: fx, fy, cx, cy, this have higher priority, can overwrite the one in config file') #相机内参
    parser.add_argument('--mode', type=str,
                        help='slam mode: mono, rgbd or stereo')
    # 解析命令行参数
    args = parser.parse_args()

    # 设置多进程启动方式为'spawn'
    torch.multiprocessing.set_start_method('spawn')

    # 加载配置文件
    cfg = config.load_config(
        args.config, './configs/go_slam.yaml'
    )

    # 根据命令行参数修改配置
    if args.mode is not None:
        cfg['mode'] = args.mode # SLAM模式，可选项为'mono', 'rgbd', 'stereo'
    if args.only_tracking:
        cfg['only_tracking'] = True # 是否只进行跟踪
    if args.image_size is not None:
        cfg['cam']['H'], cfg['cam']['W'] = args.image_size # 图像尺寸，高度和宽度
    if args.calibration_txt is not None:
        cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy'] = np.loadtxt(args.calibration_txt).tolist() # 相机内参

    # 检查SLAM模式是否正确
    assert cfg['mode'] in ['rgbd', 'mono', 'stereo'], cfg['mode']
    print(f"\n\n** Running {cfg['data']['input_folder']} in {cfg['mode']} mode!!! **\n\n")

    # 打印命令行参数
    print(args)

    # 设置输出目录
    if args.output is None:
        output_dir = cfg['data']['output']
    else:
        output_dir = args.output

    # 备份源代码
    backup_source_code(os.path.join(output_dir, 'code'))
    # 保存配置
    config.save_config(cfg, f'{output_dir}/cfg.yaml')

    # 获取数据集
    dataset = get_dataset(cfg, args, device=args.device)

    # 创建SLAM对象
    slam = SLAM(args, cfg)
    # 运行SLAM
    slam.run(dataset)

    # 结束SLAM
    slam.terminate(rank=-1, stream=dataset)

    print('Done!')


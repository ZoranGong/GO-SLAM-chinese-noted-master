# 导入所需的库
import os
import numpy as np
import torch
import torch.nn as nn
from colorama import Fore, Style
from collections import OrderedDict
from tqdm import tqdm
from lietorch import SE3
from time import gmtime, strftime, time, sleep
import torch.multiprocessing as mp

# 导入项目中的模块
from .droid_net import DroidNet
from .frontend import Frontend
from .backend import Backend
from .depth_video import DepthVideo
from .motion_filter import MotionFilter
from .multiview_filter import MultiviewFilter
from .visualization import droid_visualization
from .trajectory_filler import PoseTrajectoryFiller
from .mapping import Mapper
from .render import Renderer
from .mesher import Mesher
from .InstantNeuS import InstantNeuS

# 定义跟踪器类
class Tracker(nn.Module):
    def __init__(self, cfg, args, slam):
        super(Tracker, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = args.device
        self.net = slam.net
        self.video = slam.video
        self.verbose = slam.verbose

        # 过滤器，用于检查是否有足够的运动
        self.frontend_window = cfg['tracking']['frontend']['window']
        filter_thresh = cfg['tracking']['motion_filter']['thresh']
        self.motion_filter = MotionFilter(self.net, self.video, thresh=filter_thresh, device=self.device)

        # 前端处理
        self.frontend = Frontend(self.net, self.video, self.args, self.cfg)

    def forward(self, timestamp, image, depth, intrinsic, gt_pose=None):
        with torch.no_grad():
            # 跟踪函数
            self.motion_filter.track(timestamp, image, depth, intrinsic, gt_pose=gt_pose)

            # 局部BA
            self.frontend()

# 定义全局BA类
class BundleAdjustment(nn.Module):
    def __init__(self, cfg, args, slam):
        super(BundleAdjustment, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = args.device
        self.net = slam.net
        self.video = slam.video
        self.verbose = slam.verbose
        self.frontend_window = cfg['tracking']['frontend']['window'] #滑动窗口大小
        self.last_t = -1
        self.ba_counter = -1

        # 后端处理
        self.backend = Backend(self.net, self.video, self.args, self.cfg)

    def info(self, msg):
        print(Fore.GREEN)
        print(msg)
        print(Style.RESET_ALL)

    def forward(self):
        cur_t = self.video.counter.value # 当前帧
        t = cur_t

        if cur_t > self.frontend_window:
            t_start = 0
            now = f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} - Full BA'
            msg = f'\n\n {now} : [{t_start}, {t}]; Current Keyframe is {cur_t}, last is {self.last_t}.'

            self.backend.dense_ba(t_start=t_start, t_end=t, steps=6, motion_only=False) # 全局BA
            self.info(msg+'\n')

            self.last_t = cur_t # 更新上一帧

# 定义SLAM类
class SLAM:
    def __init__(self, args, cfg):
        super(SLAM, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = args.device #cuda:0
        self.verbose = cfg['verbose'] #True, 是否打印信息
        self.mode = cfg['mode'] #SLAM模式，可选项为'mono', 'rgbd', 'stereo'
        self.only_tracking = cfg['only_tracking'] #是否只进行跟踪
        self.make_video = args.make_video #是否生成视频

        if args.output is None:
            self.output = cfg['data']['output'] #输出目录
        else:
            self.output = args.output
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(f'{self.output}/logs/', exist_ok=True) #日志目录

        self.update_cam(cfg) # 更新相机内参
        self.load_bound(cfg) # 加载场景边界参数

        self.mapping_net = InstantNeuS(
            cfg['mapping']['model'],
            bound=cfg['mapping']['bound'],
            device=cfg['mapping']['device']
        ).to(cfg['mapping']['device']) # 初始化建图网络，InstantNeuS
        self.net = DroidNet() # 初始化跟踪网络，DroidNet

        self.load_pretrained(cfg['tracking']['pretrained']) # 加载预训练参数，./pretrained/droid.pth
        self.net.to(self.device).eval() # 将跟踪网络移动到GPU上并设置为评估模式
        self.net.share_memory() # 在进程之间共享内存
        self.mapping_net.share_memory() # 在进程之间共享内存

        self.renderer = Renderer(cfg, args, self) # 初始化渲染器

        self.num_running_thread = torch.zeros((1)).int() # 运行的线程数，初始化为0，用于进程间共享
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int() # 是否所有线程都已经触发，并放入共享内存
        self.all_trigered.share_memory_()
        self.tracking_finished = torch.zeros((1)).int() # 跟踪是否完成，放入共享内存
        self.tracking_finished.share_memory_()
        self.mapping_finished = torch.zeros((1)).int() # 建图是否完成，放入共享内存
        self.mapping_finished.share_memory_()
        self.meshing_finished = torch.zeros((1)).int() # 网格化是否完成，放入共享内存
        self.meshing_finished.share_memory_()
        self.optimizing_finished = torch.zeros((1)).int() # 优化是否完成，放入共享内存
        self.optimizing_finished.share_memory_()
        self.visualizing_finished = torch.zeros((1)).int() # 可视化是否完成，放入共享内存
        self.visualizing_finished.share_memory_()

        self.hang_on = torch.zeros((1)).int() # 是否挂起，放入共享内存
        self.hang_on.share_memory_()

        self.reload_map = torch.zeros((1)).int() # 是否重新加载地图，放入共享内存
        self.reload_map.share_memory_()
        self.post_processing_iters = cfg['mapping']['post_processing_iters'] # 后处理迭代次数，10

        # 存储图像，深度，姿态，内参（在进程之间共享）
        self.video = DepthVideo(cfg, args) # 初始化视频对象(关键帧)

        self.tracker = Tracker(cfg, args, self) # 初始化跟踪器
        self.ba = BundleAdjustment(cfg, args, self) # 初始化全局BA

        self.multiview_filter = MultiviewFilter(cfg, args, self) # 初始化多视图过滤器

        # 后处理器 - 填充非关键帧的姿态
        self.traj_filler = PoseTrajectoryFiller(net=self.net, video=self.video, device=self.device)

        self.mapper = Mapper(cfg, args, self) # 初始化建图器
        self.mesher = Mesher(cfg, args, self) # 初始化网格化器

    # 更新相机内参
    def update_cam(self, cfg):
        """
            更新相机内参，根据预处理配置，如resize或edge crop
            Update the camera intrinsics according to the pre-processing config,
            such as resize or edge crop
        """
        # resize the input images to crop_size(variable name used in lietorch)
        H, W = cfg['cam']['H'], cfg['cam']['W'] # 图像尺寸，高度和宽度
        fx, fy = cfg['cam']['fx'], cfg['cam']['fy'] # 相机内参
        cx, cy = cfg['cam']['cx'], cfg['cam']['cy']

        h_edge, w_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']
        H_out, W_out = cfg['cam']['H_out'], cfg['cam']['W_out']

        #根据新的图像大小来更新焦距。新的焦距（self.fx, self.fy）是通过将原始焦距与新图像大小
        # （包括边缘）与原图像大小的比例进行缩放来计算的
        self.fx = fx * (W_out + w_edge * 2) / W
        self.fy = fy * (H_out + h_edge * 2) / H
        self.cx = cx * (W_out + w_edge * 2) / W
        self.cy = cy * (H_out + h_edge * 2) / H
        self.H, self.W = H_out, W_out

        self.cx = self.cx - w_edge
        self.cy = self.cy - h_edge
        # update_cam方法根据预处理配置来调整相机的内参，这包括调整图像的大小和裁剪边缘。

    # 加载场景边界参数
    def load_bound(self, cfg):
        """
            # 根据不同的解码器，将场景边界参数传递给不同的解码器和自身
            Pass the scene bound parameters to different decoders and self.
            Args:
            cfg:                        (dict), parsed config dict
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(
            np.array(cfg['mapping']['bound'])
        ).float()

    def load_pretrained(self, pretrained): # 加载预训练参数
        print(f'INFO: load pretrained checkpiont from {pretrained}!')

        state_dict = OrderedDict([
            (k.replace('module.', ''), v) for (k, v) in torch.load(pretrained).items()
        ])

        state_dict['update.weight.2.weight'] = state_dict['update.weight.2.weight'][:2]
        state_dict['update.weight.2.bias'] = state_dict['update.weight.2.bias'][:2]
        state_dict['update.delta.2.weight'] = state_dict['update.delta.2.weight'][:2]
        state_dict['update.delta.2.bias'] = state_dict['update.delta.2.bias'][:2]

        self.net.load_state_dict(state_dict)

    # 跟踪
    def tracking(self, rank, stream):
        print('Tracking Triggered!')
        self.all_trigered += 1 # 线程触发数加1
        while (self.all_trigered < self.num_running_thread):
            pass # 等待所有线程都触发
        for (timestamp, image, depth, intrinsic, gt_pose) in tqdm(stream):
            if self.mode != 'rgbd':
                depth = None
            self.tracker(timestamp, image, depth, intrinsic, gt_pose) # 开始跟踪

            # predict mesh every 50 frames for video making，每50帧预测一次网格以制作视频
            if timestamp % 50 == 0 and timestamp > 0 and self.make_video:
                self.hang_on[:] = 1 # 挂起，等待网格化完成
            while (self.hang_on > 0):
                sleep(1.0)

        self.tracking_finished += 1 # 跟踪完成数加1
        print('Tracking Done!')

    # 优化
    def optimizing(self, rank, dont_run=False):
        print('Full Bundle Adjustment Triggered!')
        self.all_trigered += 1 # 线程触发数加1
        while (self.tracking_finished < 1 and not dont_run): # 跟踪结束前就一直优化BA
            while (self.hang_on > 0 and self.make_video): # 等待网格化完成
                sleep(1.0)
            self.ba() # 等待网格化完成后，开始全局BA

        if not dont_run:
            self.ba()
        self.optimizing_finished += 1 # 优化完成数加1

        print('Full Bundle Adjustment Done!')

    # 多视图过滤
    def multiview_filtering(self, rank, dont_run=False):
        print('Multiview Filtering Triggered!') # 多视图过滤触发
        self.all_trigered += 1
        while ((self.tracking_finished < 1 or self.optimizing_finished < 1) and not dont_run):
            while (self.hang_on > 0 and self.make_video):
                sleep(1.0)
            self.multiview_filter() # 一样，等待网格化完成后，开始多视图过滤

        print('Multiview Filtering Done!')

    # 建图
    def mapping(self, rank, dont_run=False):
        print('Dense Mapping Triggered!')
        self.all_trigered += 1
        while (self.tracking_finished < 1 and not dont_run):
            while (self.hang_on > 0 and self.make_video):
                sleep(1.0)
            self.mapper() # 一样，等待网格化完成后，开始建图，只要跟踪没有完成，就一直可以建图

        if not dont_run:
            print('Start post-processing on mapping...')
            for i in tqdm(range(self.post_processing_iters)):
                self.mapper(the_end=True)
        self.mapping_finished += 1
        print('Dense Mapping Done!')

    # 网格化
    def meshing(self, rank, dont_run=False):
        print('Meshing Triggered!')
        self.all_trigered += 1
        while (self.mapping_finished < 1 and (not dont_run)):
            while (self.hang_on < 1 and self.mapping_finished < 1 and self.make_video):
                sleep(1.0)
            self.mesher() # 等挂起的时候，开始网格化，只要建图没有完成，就一直可以网格化
            self.hang_on[:] = 0

        self.meshing_finished += 1
        print('Meshing Done!')

    # 可视化
    def visualizing(self, rank, dont_run=False):
        print('Visualization Triggered!')
        self.all_trigered += 1
        while ((self.tracking_finished < 1 or self.optimizing_finished < 1) and (not dont_run)):
            droid_visualization(self.video, device=self.device, save_root=self.output)

        self.visualizing_finished += 1
        print('Visualization Done!')

    # 终止
    def terminate(self, rank, stream=None):
        # 填充非关键帧的姿态并评估
        """ fill poses for non-keyframe images and evaluate """

        while (self.optimizing_finished < 1):
            if self.num_running_thread == 1 and self.tracking_finished > 0:
                break

        os.makedirs(f'{self.output}/checkpoints/', exist_ok=True)
        torch.save({
            'mapping_net': self.mapping_net.state_dict(),
            'tracking_net': self.net.state_dict(),
            'keyframe_timestamps': self.video.timestamp,
        }, f'{self.output}/checkpoints/go.ckpt')

        do_evaluation = True
        if do_evaluation: #用evo包评估
            from evo.core.trajectory import PoseTrajectory3D
            import evo.main_ape as main_ape
            from evo.core.metrics import PoseRelation
            from evo.core.trajectory import PosePath3D
            import numpy as np

            print("#" * 20 + f" Results for {stream.input_folder} ...")

            timestamps = [i for i in range(len(stream))]
            camera_trajectory = self.traj_filler(stream)  # w2cs
            w2w = SE3(self.video.pose_compensate[0].clone().unsqueeze(dim=0)).to(camera_trajectory.device)
            camera_trajectory = w2w * camera_trajectory.inv()
            traj_est = camera_trajectory.data.cpu().numpy()
            estimate_c2w_list = camera_trajectory.matrix().data.cpu()
            np.save(
                f'{self.output}/checkpoints/est_poses.npy',
                estimate_c2w_list.numpy(),  # c2ws
            )

            traj_ref = []
            traj_est_select = []
            if stream.poses is None:  # for eth3d submission
                if stream.image_timestamps is not None:
                    submission_txt = f'{self.output}/submission.txt'
                    with open(submission_txt, 'w') as fp:
                        for tm, pos in zip(stream.image_timestamps, traj_est.tolist()):
                            str = f'{tm:.9f}'
                            for ps in pos:  # timestamp tx ty tz qx qy qz qw
                                str += f' {ps:.14f}'
                            fp.write(str + '\n')
                    print('Poses are save to {}!'.format(submission_txt))

                print("Terminate: no GT poses found!")
                trans_init = None
                gt_c2w_list = None
            else:
                for i in range(len(stream.poses)):
                    val = stream.poses[i].sum()
                    if np.isnan(val) or np.isinf(val):
                        print(f'Nan or Inf found in gt poses, skipping {i}th pose!')
                        continue
                    traj_est_select.append(traj_est[i])
                    traj_ref.append(stream.poses[i])

                traj_est = np.stack(traj_est_select, axis=0)
                gt_c2w_list = torch.from_numpy(np.stack(traj_ref, axis=0))

                traj_est = PoseTrajectory3D(
                    positions_xyz=traj_est[:, :3],
                    orientations_quat_wxyz=traj_est[:, 3:],
                    timestamps=np.array(timestamps))

                traj_ref = PosePath3D(poses_se3=traj_ref)

                result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                                      pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

                out_path = f'{self.output}/metrics_traj.txt'
                with open(out_path, 'a') as fp:
                    fp.write(result.pretty_str())
                trans_init = result.np_arrays['alignment_transformation_sim3']

            if self.meshing_finished > 0 and (not self.only_tracking):
                self.mesher(the_end=True, estimate_c2w_list=estimate_c2w_list, gt_c2w_list=gt_c2w_list,
                            trans_init=trans_init)

        print("Terminate: Done!")

    # 运行
    def run(self, stream):
        dont_run = True
        processes = [
            mp.Process(target=self.tracking, args=(0, stream)),
            mp.Process(target=self.optimizing, args=(1, not dont_run)),
            mp.Process(target=self.multiview_filtering, args=(2, dont_run if self.only_tracking else (not dont_run))),
            mp.Process(target=self.mapping, args=(3, dont_run if self.only_tracking else (not dont_run))),
            mp.Process(target=self.meshing, args=(4, dont_run)),  # only generate mesh at the very end
            # mp.Process(target=self.meshing, args=(4, dont_run if self.only_tracking else (not dont_run))),
            mp.Process(target=self.visualizing, args=(5, dont_run)),
        ]

        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start() #全部启动！！

        for p in processes:
            p.join() #join方法会阻塞当前进程，直到p进程执行完毕。这样可以确保主进程在所有子进程都完成任务后再继续执行
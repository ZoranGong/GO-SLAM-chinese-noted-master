import torch
from copy import deepcopy
from time import gmtime, strftime, time

from .factor_graph import FactorGraph
from .backend import Backend as LoopClosing


class Frontend:
    def __init__(self, net, video, args, cfg):
        self.video = video
        self.update_op = net.update
        self.warmup = cfg['tracking']['warmup'] # 8, warmup帧之后才开始进行初始化
        self.upsample = cfg['tracking']['upsample'] # True, 是否进行上采样
        self.beta = cfg['tracking']['beta'] # 0.75, 用于计算相关性体积的beta值，
        self.verbose = cfg['verbose'] # True, 是否打印详细信息

        self.frontend_max_factors = cfg['tracking']['frontend']['max_factors'] # 75, 局部BA中可优化的最大边数
        self.frontend_nms = cfg['tracking']['frontend']['nms'] # 1, 非极大值抑制的半径
        self.keyframe_thresh = cfg['tracking']['frontend']['keyframe_thresh'] #4.0, 用于判断是否可剔除关键帧的阈值，小于该阈值则剔除
        self.frontend_window = cfg['tracking']['frontend']['window'] # 25, 局部优化窗口的大小
        self.frontend_thresh = cfg['tracking']['frontend']['thresh'] # 16, 只考虑平均光流小于该阈值的边进行优化
        self.frontend_radius = cfg['tracking']['frontend']['radius'] # 1, 局部优化窗口的半径
        self.enable_loop = cfg['tracking']['frontend']['enable_loop'] # True, 是否进行闭环检测
        self.loop_closing = LoopClosing(net, video, args, cfg) # 闭环检测对象
        self.last_loop_t = -1

        self.graph = FactorGraph(
            video, net.update,
            device=args.device,
            corr_impl='volume',
            max_factors=self.frontend_max_factors,
            upsample=self.upsample
        ) # 初始化关键帧图对象

        # local optimization window，局部优化窗口
        self.t0 = 0 # 局部优化窗口左边界
        self.t1 = 0 # 局部优化窗口右边界

        # frontend variables，前端变量
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

    def __update(self):
        # 添加边，进行更新
        """ add edges, perform update """

        self.count += 1 # 帧计数
        self.t1 += 1 #滑动窗口右边界 + 1

        if self.graph.corr is not None: # 如果相关性体积不为空
            self.graph.rm_factors(self.graph.age > self.max_age, store=True) # 只取最近的25帧，max_age = 25(论文里为N_local)

        # build edges between [t1-5, video.counter] and [t1-window, video.counter],在[t1-5, video.counter]和[t1-window, video.counter]之间建立边
        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0),
                                         rad=self.frontend_radius, nms=self.frontend_nms,
                                         thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0,
                                                  self.video.disps_sens[self.t1-1],
                                                  self.video.disps[self.t1-1])

        for itr in range(self.iters1):
            self.graph.update(t0=None, t1=None, use_inactive=True) # 更新图

        # set initial pose for next frame，设置下一帧的初始位姿
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True) # 计算两帧之间的距离

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1-2) # 如果距离（光流）小于阈值，则剔除关键帧，要确保关键帧之间的距离足够大

            with self.video.get_lock():
                self.video.counter.value -= 1 # 帧计数 - 1
                self.t1 -= 1 # 滑动窗口右边界 - 1
        else:
            cur_t = self.video.counter.value # 当前帧计数
            t_start = 0  # 局部优化窗口左边界
            now = f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} - Loop BA'
            msg = f'\n\n {now} : [{t_start}, {cur_t}]; Current Keyframe is {cur_t}, last is {self.last_loop_t}.'
            if self.enable_loop and cur_t > self.frontend_window: # 如果启用闭环检测且有足够的关键帧
                n_kf, n_edge = self.loop_closing.loop_ba(t_start=0, t_end=cur_t, steps=self.iters2, motion_only=False, local_graph=self.graph)
                # 启动闭环检测，进行局部优化，输出优化的关键帧数和边数
                print(msg + f' {n_kf} KFs, last KF is {self.last_loop_t}! \n')
                self.last_loop_t = cur_t # 更新上一次的关键帧

            else:
                for itr in range(self.iters2):
                    self.graph.update(t0=None, t1=None, use_inactive=True) # 更新图

        # set pose for next iteration，设置下一次迭代的位姿
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization，更新可视化
        self.video.dirty[self.graph.ii.min():self.t1] = True

    def __initialize(self):
        # 初始化
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value # 滑动窗口右边界

        # build edges between nearby(radius <= 3) frames within local windown [t0, t1]
        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3) # 在局部窗口[t0, t1]内建立边，半径为3，即相邻帧之间建立边

        for itr in range(8):
            self.graph.update(t0=1, t1=None, use_inactive=True) # 迭代8次，更新关键帧图

        # build edges between [t0, video.counter] and [t1, video.counter]
        self.graph.add_proximity_factors(t0=0, t1=0, rad=2, nms=2,
                                         thresh=self.frontend_thresh,
                                         remove=False) # 在[t0, video.counter]和[t1, video.counter]之间建立边，半径为2，非极大值抑制半径为2，阈值为16

        for itr in range(8):
            self.graph.update(t0=1, t1=None, use_inactive=True) # 迭代8次，更新关键帧图

        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()


        # initialization complete，初始化完成
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone() # 上一帧的位姿
        self.last_disp = self.video.disps[self.t1-1].clone() # 上一帧的光流
        self.last_time = self.video.timestamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True) # 删除前4帧的边（不知道为什么）

    def __call__(self):
        """ main update """

        # do initialization，进行初始化
        if not self.is_initialized and self.video.counter.value == self.warmup: # 如果帧计数等于warmup帧数
            self.__initialize() # 进行初始化

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value: # 如果已经初始化且滑动窗口右边界小于帧计数
            self.__update() # 进行更新

        else:
            pass





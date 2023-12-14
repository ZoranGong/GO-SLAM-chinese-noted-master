import os
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
from colorama import Fore, Style
from torch.autograd import Variable
from .Logger import TextLogger
from .nerf_func import random_select, build_rays, Rt_to_quaternion, quaternion_to_Rt


class Mapper(object):
    """
    Mpper thread.
    """

    def __init__(self, cfg, args, slam):
        self.cfg = cfg
        self.args = args
        self.verbose = slam.verbose

        self.bound = slam.bound
        self.video = slam.video
        self.mapping_net = slam.mapping_net # 建图网络
        self.renderer = slam.renderer # 渲染器
        self.reload_map = slam.reload_map # 重建地图的次数

        self.output = slam.output

        self.device = cfg['mapping']['device']
        self.num_joint_iters = cfg['mapping']['iters'] # 10，每次优化的迭代次数
        self.decay = float(cfg['mapping']['decay']) # 0.8，每次优化的学习率衰减系数
        self.w_color_loss = cfg['mapping']['w_color_loss'] # 2.0，颜色损失的权重
        self.w_sdf_loss = cfg['mapping']['w_sdf_loss'] # 2.0 ，SDF损失的权重
        self.w_eikonal_loss = cfg['mapping']['w_eikonal_loss'] # 0.1，Eikonal损失的权重
        self.uncertainty_based = cfg['mapping']['uncertainty_weight_loss'] # True，是否使用不确定性加权

        self.BA = cfg['mapping']['BA']  # Even if BA is enabled, it starts only when there are at least 4 keyframes
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr'] # 0.001，相机位姿优化的学习率
        self.mapping_pixels = cfg['mapping']['pixels'] # 4400，每次优化的像素数
        self.mapping_window_size = cfg['mapping']['mapping_window_size'] # 22，每次优化的关键帧数

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.local_step = 0 # 本地优化的步数
        self.global_step = 0 # 全局优化的步数
        self.last_visit = 0 # 上一次优化的关键帧
        self.init = True

        os.makedirs(f'{self.output}/logs/mapping/', exist_ok=True)
        self.txt = TextLogger(f'{self.output}/logs/mapping/log.txt') # 日志

        ignore_keys = ()
        net_param = self.mapping_net.get_training_parameters(ignore_keys=ignore_keys) # 获取网络训练参数
        grid_param = self.mapping_net.get_volume_parameters()
        self.train_params = list(net_param) + list(grid_param)
        self.optimizer = torch.optim.AdamW([
            {'params': net_param, 'lr': cfg['mapping']['net_lr']},
            {'params': grid_param, 'lr': cfg['mapping']['grid_lr']},
        ], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01) #把网络参数和体素网格参数分开，放入Adam

    def optimize_map(self,
                     rays_o,
                     rays_d,
                     rays_color,
                     rays_depth,
                     optimizer,
                     num_joint_iters):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enables).
        建图迭代，从选定的关键帧中采样像素，然后优化场景表示和相机位姿（如果启用局部BA）。
        Args:
            num_joint_iters:            (int), number of mapping iterations.，建图迭代次数
            lr_factor:                  (float), current_lr * lr_factor，学习率衰减系数
            writer:                     Tensorboard SummaryWriter，可视化工具

        Returns:
            cur_c2w/None:               (Tensor), the updated cur_c2w, return None if no BA
        """
        device = self.device

        for joint_iter in range(1, num_joint_iters+1):
            self.local_step += 1
            self.global_step += 1

            optimizer.zero_grad()

            render_params = {
                'global_step': self.global_step,
            }
            # 下面的render_batch_ray渲染是NICE-SLAM的代码
            with torch.enable_grad():
                ret = self.renderer.render_batch_ray(rays_o=rays_o, rays_d=rays_d, net=self.mapping_net.to(device),
                                                     render_params=render_params, device=device, gt_depth=rays_depth) # 渲染

            rays_depth = rays_depth.reshape(-1, 1)  # [n_rays, 1]
            valid_mask = (rays_depth > 0).reshape(-1)  # [n_rays, ]

            rays_depth = rays_depth[valid_mask]
            rays_color = rays_color[valid_mask]
            est_color = ret['color'][valid_mask]  # [n_rays, 3]
            est_depth = ret['depth'][valid_mask]  # [n_rays, 1]
            sdf = ret['sdf'][valid_mask]  # [n_rays, n_samples]
            z_vals = ret['z_vals'][valid_mask]  # [n_rays, n_samples]
            depth_variance = ret['depth_variance'][valid_mask] # [n_rays, 1]
            gradient_error = ret['gradient_error']  # [1, ]
            uncertainty_weight = 1.0 / torch.sqrt(depth_variance.detach() + 1e-10) # [n_rays, 1]
            if not self.uncertainty_based:
                uncertainty_weight = torch.ones_like(uncertainty_weight)

            assert rays_depth.shape == est_depth.shape, f'{rays_depth.shape}, {est_depth.shape}!'

            total_loss = 0.0

            # -- color loss --
            color_loss = torch.abs(est_color - rays_color).mean()
            total_loss = total_loss + color_loss * self.w_color_loss

            # -- depth loss --
            mae = torch.abs(est_depth - rays_depth)
            depth_loss = (mae * uncertainty_weight).mean()
            total_loss = total_loss + depth_loss * 1.0

            # -- sdf loss --
            sdf_loss, sparse_loss = 0.0, 0.0
            if self.w_sdf_loss > 0:
                sdf_loss, sparse_loss = self.mapping_net.compute_sdf_error(sdf=sdf, z_vals=z_vals, gt_depth=rays_depth)
                total_loss = total_loss + (sdf_loss + sparse_loss) * self.w_sdf_loss

            # -- eikonal loss --
            if self.w_eikonal_loss > 0:
                eikonal_loss = gradient_error.mean()
                total_loss = total_loss + self.w_eikonal_loss * eikonal_loss

            total_loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.train_params, max_norm=35.0) # 梯度裁剪，防止梯度爆炸
            optimizer.step() # 更新参数
            optimizer.zero_grad() # 清空梯度

            if (self.local_step % self.num_joint_iters == 0) and self.verbose: # 每10次迭代打印一次日志msg
                msg = ''
                list_lr = []
                for g in optimizer.param_groups:
                    list_lr.append(round(g['lr'], 6))
                msg += 'Lr : {}'.format(list_lr)
                msg += f' | Loss of total: {total_loss.detach():.4f}, depth: {depth_loss:.4f}, ' \
                       f'color: {color_loss:.4f}, ' \
                       f'sdf: {sdf_loss:.4f}, n_rays: {rays_o.shape}!'
                self.txt.info(msg)


    def __call__(self, the_end=False): # 未访问的关键帧unvisited和访问的关键帧visited分别依次进行渲染和优化
        cur_idx = int(self.video.filtered_id.item())  # valid idx [0, 1, ..., cur_idx-1]，当前帧的索引
        if cur_idx > 1:
            # cur_idx = min(cur_idx, self.last_visit+per_keyframe)
            timestamp = self.video.timestamp[cur_idx-1] # 当前帧的时间戳
            num_joint_iters = self.num_joint_iters  # 10，每次优化的迭代次数
            if the_end:
                num_joint_iters = num_joint_iters * 10 # 如果是最后一帧，则迭代次数乘以10
            device = self.device
            self.local_step = 0

            unvisit_list = list(range(self.last_visit, cur_idx)) #未访问的关键帧，范围是上一次优化的关键帧到当前帧
            visit_list = [cur_idx-1, cur_idx-2] #最新的两帧确保被访问
            if self.last_visit > 0:
                priority = self.video.update_priority[:self.last_visit].detach()
                _, indices = torch.sort(priority, dim=0, descending=True) # 姿态差异降序排列
                indices = list(indices.cpu().numpy())
                visit_list += indices[:10] #访问时从排序列表中选择前 10 个关键帧。
                visit_list += random_select(self.last_visit, self.mapping_window_size-12) # 随机选择10个关键帧
            #论文原话：按照当前状态和上次更新状态之间的姿态差异降序对所有关键帧进行排序，并在访问时从排序列表中选择前 10 个关键帧。
            visit_frame = {}
            visit_ba_list = []
            enable_ba = ((self.BA) and (self.last_visit >= 10)) #距离上次优化的关键帧间隔大于10帧时，则再次进行BA优化
            for frame_id, frame in enumerate(visit_list):
                frame_items = self.video.get_mapping_item(frame, device, decay=self.decay) # 获取已访问关键帧的颜色、深度、相机位姿、相机位姿的四元数、mask
                visit_frame[frame] = frame_items
                if enable_ba:
                    _, _, c2w_mat, _, _ = frame_items
                    quadt = Rt_to_quaternion(c2w_mat, Tquad=False) # 相机位姿的四元数
                    quadt = Variable(quadt.to(self.device), requires_grad=True)
                    visit_ba_list.append(quadt) # 添加到visit_ba_list中


            unvisit_frame = {}
            for frame in unvisit_list:
                unvisit_frame[frame] = self.video.get_mapping_item(frame, device, decay=self.decay) # 获取未访问关键帧的颜色，位姿等信息

            H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

            optimizer = self.optimizer
            if enable_ba and len(optimizer.param_groups) > 2:  # Attention the number 2 set here，
                del optimizer.param_groups[-1] # 删除最后一个参数组，确保只有两个参数组，一个是网络参数，一个是体素网格参数
            if enable_ba and len(visit_ba_list) > 0:
                optimizer.add_param_group({'params': visit_ba_list, 'lr': self.BA_cam_lr}) # 添加相机位姿参数组，位姿和地图为一起优化

            bd = self.video.get_bound()
            with self.video.mapping.get_lock():
                self.mapping_net.update_bound(bd) # 更新地图的边界

            prefix = f"Bound: ["
            bd = self.mapping_net.realtime_bound.tolist() # 获取地图的边界
            prefix += f'[{bd[0][0]:.1f}, {bd[0][1]:.1f}], '
            prefix += f'[{bd[1][0]:.1f}, {bd[1][1]:.1f}], '
            prefix += f'[{bd[2][0]:.1f}, {bd[2][1]:.1f}]]; '
            print(Fore.MAGENTA)
            if self.verbose:
                self.txt.info(prefix + 'Mapping Frame {}, unvisit {}, has visited {}'.format(timestamp.item(), unvisit_list, visit_list)) # 打印日志
            else:
                if len(unvisit_list) > 2:
                    self.txt.info(prefix + 'Mapping Frame {}, unvisit kf are: {}!'.format(timestamp.item(), unvisit_list))
            print(Style.RESET_ALL)


            # unvisit，未访问的关键帧
            unvisit_factor = num_joint_iters * 10 if self.init else num_joint_iters # 如果是第一次建图，则迭代次数乘以10
            if len(unvisit_list) > 2:
                self.last_visit = cur_idx # 更新上一次优化的关键帧last_visit
                for _ in range(unvisit_factor):  #开始建图迭代
                    unvisit_rays_d = []
                    unvisit_rays_o = []
                    unvisit_gt_depth = []
                    unvisit_gt_color = []

                    sub_unvisit_list = list(np.random.choice(unvisit_list, self.mapping_window_size))
                    n_rays_unvisit = self.mapping_pixels // len(sub_unvisit_list)
                    for frame_idx, frame in enumerate(sub_unvisit_list):
                        gt_color, gt_depth, c2w, gt_c2w, mask = unvisit_frame[frame]
                        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = build_rays(
                            0, H, 0, W, n_rays_unvisit, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device,
                            nerf_coordinate=False, dir_normalize=False, mask=mask
                        ) #通过相机位姿和深度图生成采样射线
                        unvisit_rays_o.append(batch_rays_o.float())  # visited代表已经渲染1过的关键帧，unvisit代表未渲染过的关键帧
                        unvisit_rays_d.append(batch_rays_d.float())
                        unvisit_gt_depth.append(batch_gt_depth.float())
                        unvisit_gt_color.append(batch_gt_color.float())

                    unvisit_rays_o = torch.cat(unvisit_rays_o, dim=0) # 将所有关键帧的射线拼接起来，o代表射线的起点,d代表射线的方向
                    unvisit_rays_d = torch.cat(unvisit_rays_d, dim=0)
                    unvisit_gt_depth = torch.cat(unvisit_gt_depth, dim=0)
                    unvisit_gt_color = torch.cat(unvisit_gt_color, dim=0)

                    if len(unvisit_rays_o) < 100:
                        continue

                    self.optimize_map(
                        rays_o=unvisit_rays_o,
                        rays_d=unvisit_rays_d,
                        rays_color=unvisit_gt_color,
                        rays_depth=unvisit_gt_depth,
                        optimizer=optimizer,
                        num_joint_iters=1,
                    ) # 开始渲染+优化地图，把体素网格和神经渲染网络放在一起和位姿一起优化

            torch.cuda.empty_cache()

            # visit，访问的关键帧
            for _ in range(num_joint_iters):
                if len(visit_list) < 1:
                    continue

                n_rays = self.mapping_pixels // len(visit_list)
                visit_rays_d_list = []
                visit_rays_o_list = []
                visit_gt_depth_list = []
                visit_gt_color_list = []
                for frame_id, frame in enumerate(visit_list):
                    gt_color, gt_depth, c2w, gt_c2w, mask = visit_frame[frame]
                    if enable_ba:
                        quadt = visit_ba_list[frame_id]
                        c2w = quaternion_to_Rt(quadt)
                    batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = build_rays(
                        0, H, 0, W, n_rays, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device,
                        nerf_coordinate=False, dir_normalize=False, mask=mask
                    ) # 通过相机位姿和深度图生成采样射线
                    visit_rays_o_list.append(batch_rays_o.float())
                    visit_rays_d_list.append(batch_rays_d.float()) #访问的关键帧再来一次渲染
                    visit_gt_depth_list.append(batch_gt_depth.float())
                    visit_gt_color_list.append(batch_gt_color.float())

                visit_rays_o = torch.cat(visit_rays_o_list, dim=0)
                visit_rays_d = torch.cat(visit_rays_d_list, dim=0)
                visit_gt_depth = torch.cat(visit_gt_depth_list, dim=0)
                visit_gt_color = torch.cat(visit_gt_color_list, dim=0)

                if len(visit_rays_o) < 100:
                    continue

                self.optimize_map(
                    rays_o=visit_rays_o,
                    rays_d=visit_rays_d,
                    rays_color=visit_gt_color,
                    rays_depth=visit_gt_depth,
                    optimizer=optimizer,
                    num_joint_iters=1,
                ) # 开始渲染+优化地图，把体素网格和神经渲染网络放在一起和位姿一起优化

            # 3d mesh has been updated, info the mesher to regenerate mesh
            self.reload_map += 1
            self.init = False
            torch.cuda.empty_cache()
            del visit_frame, unvisit_frame


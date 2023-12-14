import torch
import numpy as np
from copy import deepcopy
from .factor_graph import FactorGraph


class Backend:
    def __init__(self, net, video, args, cfg):
        self.video = video
        self.device = args.device
        self.update_op = net.update

        self.upsample = cfg['tracking']['upsample'] # True，是否进行上采样
        self.beta = cfg['tracking']['beta'] # 0.75，用于计算相关性体积的beta值
        self.backend_thresh = cfg['tracking']['backend']['thresh'] # 25，只考虑优化平均光流小于25像素的边
        self.backend_radius = cfg['tracking']['backend']['radius'] # 1，局部优化窗口的半径
        self.backend_nms = cfg['tracking']['backend']['nms'] # 5，非极大值抑制的半径

        self.backend_loop_window = cfg['tracking']['backend']['loop_window'] # 25，局部优化窗口的大小
        self.backend_loop_thresh = cfg['tracking']['backend']['loop_thresh'] # 25，只考虑平均光流小于50像素的边进行优化
        self.backend_loop_radius = cfg['tracking']['backend']['loop_radius'] # 1，局部优化窗口的半径
        self.backend_loop_nms = cfg['tracking']['backend']['loop_nms'] # 12，非极大值抑制的半径

    # 全局BA
    @torch.no_grad()
    def ba(self, t_start, t_end, steps, graph, nms, radius, thresh, max_factors, t_start_loop=None, loop=False, motion_only=False):
        """ main update """
        if t_start_loop is None or not loop:
            t_start_loop = t_start
        assert t_start_loop >= t_start, f'short: {t_start_loop}, long: {t_start}.'

        ilen = (t_end - t_start_loop) # 闭环检测窗口的长度
        jlen = (t_end - t_start) # 全局优化滑动窗口的长度
        ix = torch.arange(t_start_loop, t_end) # 闭环检测窗口的左边界到右边界
        jx = torch.arange(t_start, t_end) # 全局优化滑动窗口的左边界到右边界

        ii, jj = torch.meshgrid(ix, jx, indexing='ij')
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=self.beta) # 两帧之间的平均光流距离
        rawd = deepcopy(d).reshape(ilen, jlen) # 深拷贝d，用于闭环检测
        d[ii - radius < jj] = np.inf # 闭环检测窗口左边界到全局优化滑动窗口左边界之间的距离设为无穷大，回忆一下论文里的图，那个全局帧到本地帧之间没有值
        d[d > thresh] = np.inf # 平均光流距离大于阈值的设为无穷大，相当于mask
        d = d.reshape(ilen, jlen)

        es = []
        # build edges within local window [i-rad, i]，在局部优化窗口[i-rad, i]内建立边
        for i in range(t_start_loop, t_end): # i是闭环检测窗口的帧
            if self.video.stereo and not loop:
                es.append((i, i)) # 双目相机，双向边
                di, dj = i-t_start_loop, i-t_start
                d[di, dj] = np.inf # 局部优化窗口左边界到全局优化滑动窗口左边界的距离设为无穷大，相当于mask

            for j in range(max(i-radius, t_start_loop), i):  # j in [i-radius, i-1]，j是i半径内的帧
                es.append((i, j))
                es.append((j, i))
                di, dj = i-t_start_loop, j-t_start
                d[di, dj] = np.inf # 全局优化滑动窗口半径内的距离设为无穷大，相当于nms的mask
                d[max(0, di-nms):min(ilen, di+nms+1), max(0, dj-nms):min(jlen, dj+nms+1)] = np.inf # 非极大值抑制，周围的帧不会被选中，从而降低冗余

        # distance from small to big，按照距离从小到大排序
        vals, ix = torch.sort(d.reshape(-1), descending=False)
        ix = ix[vals<=thresh] # 只考虑平均光流小于阈值的边
        ix = ix.tolist()

        n_neighboring = 1
        while len(ix) > 0:
            k = ix.pop(0) # 取出第一个
            di, dj = k // jlen, k % jlen # 计算在滑动窗口的索引，jlen为全局优化滑动窗口的长度,

            if d[di, dj].item() > thresh: # 如果平均光流距离大于阈值，则跳过
                continue

            if len(es) > max_factors: # 边的数量大于最大边数，则跳出循环
                break

            i, j = ii[k], jj[k]
            # bidirectional，双向边
            if loop:
                sub_es = []
                num_loop = 0
                for si in range(max(i-n_neighboring, t_start_loop), min(i+n_neighboring+1, t_end)): # 遍历闭环检测窗口
                    for sj in range(max(j-n_neighboring, t_start), min(j+n_neighboring+1, t_end)): # 遍历全局优化滑动窗口
                        if rawd[(si-t_start_loop), (sj-t_start)] <= thresh: # 如果平均光流距离小于阈值
                            num_loop += 1 # 发现闭环
                            if si != sj:
                                sub_es += [(si, sj)] # 添加双向边，闭环检测窗口的帧与全局优化滑动窗口的帧之间建立边，即加入回环边
                if num_loop > int(((n_neighboring * 2 + 1) ** 2) * 0.5): # 如果发现的闭环数量大于某特定计算值，则添加双向边
                    es += sub_es
            else:
                es += [(i, j), ] #添加双向边
                es += [(j, i), ]

            d[max(0, di-nms):min(ilen, di+nms+1), max(0, dj-nms):min(jlen, dj+nms+1)] = np.inf # 非极大值抑制，周围的帧不会被选中，从而降低冗余

        if len(es) < 3:
            return 0

        ii, jj = torch.tensor(es, device=self.device).unbind(dim=-1)

        graph.add_factors(ii, jj, remove=True) #添加BA优化后的新边

        edge_num = len(graph.ii) #更新边的数量

        #下面运行关键帧图的更新操作，这个操作消耗的内存比较小
        graph.update_lowmem( #修复起点以避免漂移，请务必在此处使用t_start_loop而不是t_start。
            t0=t_start_loop+1,  # fix the start point to avoid drift, be sure to use t_start_loop rather than t_start here.
            t1=t_end,
            iters=2,
            use_inactive=False,
            steps=steps,
            max_t=t_end,
            ba_type='dense',
            motion_only=motion_only,
        )


        graph.clear_edges()

        torch.cuda.empty_cache()

        self.video.dirty[t_start:t_end] = True

        return edge_num # 返回边的数量

    # 稠密图的BA
    @torch.no_grad()
    def dense_ba(self, t_start, t_end, steps=6, motion_only=False):
        nms = self.backend_nms
        radius = self.backend_radius
        thresh = self.backend_thresh
        n = t_end - t_start
        max_factors = (int(self.video.stereo) + (radius + 2) * 2) * n

        graph = FactorGraph(self.video, self.update_op, device=self.device, corr_impl='alt', max_factors=max_factors, upsample=self.upsample) # 初始化关键帧图对象
        n_edges = self.ba(t_start, t_end, steps, graph, nms, radius, thresh, max_factors, motion_only=motion_only) # 进行BA优化，返回边的数量

        del graph

        return n, n_edges

    # 闭环检测
    @torch.no_grad()
    def loop_ba(self, t_start, t_end, steps=6, motion_only=False, local_graph=None):
        radius = self.backend_radius
        window = self.backend_loop_window
        max_factors = 8 * window
        nms = self.backend_loop_nms
        thresh = self.backend_loop_thresh
        t_start_loop = max(0, t_end - window)

        graph = FactorGraph(self.video, self.update_op, device=self.device, corr_impl='alt', max_factors=max_factors, upsample=self.upsample) # 初始化关键帧图对象
        if local_graph is not None:
            copy_attr = ['ii', 'jj', 'age', 'net', 'target', 'weight']
            for key in copy_attr:
                val = getattr(local_graph, key)
                if val is not None:
                    setattr(graph, key, deepcopy(val)) #将局部图的属性复制到进创建的graph中

        left_factors = max_factors - len(graph.ii)
        n_edges = self.ba(t_start, t_end, steps, graph, nms, radius, thresh, left_factors, t_start_loop=t_start_loop, loop=True, motion_only=motion_only) # 进行BA优化，返回边的数量

        del graph

        return t_end - t_start_loop, n_edges # 返回闭环检测窗口的长度和边的数量

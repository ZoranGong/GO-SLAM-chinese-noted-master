import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import mcubes
import trimesh
import numpy as np


def normalized_3d_coordinate(p, bound): # p是3D点，bound是场景边界，归一化
    """
    归一化坐标到[-1, 1]，对应给定的边界框
    Normalize coordinate to [-1, 1], corresponds to the bounding box given
    Args:
        p:                              (Tensor), coordiate in 3d space
                                        [N, 3], 3: x, y, z
        bound:                          (Tensor), the scene bound
                                        [3, 2], 3: x, y, z

    Returns:
        p:                              (Tensor), normalized coordiate in 3d space
                                        [N, 3]

    """

    p = p.reshape(-1, 3)
    bound = bound.to(p.device)
    p = (p - bound[:, 0]) / (bound[:, 1] - bound[:, 0]) * 2.0 - 1.0
    p = p.clamp(min=-1.0, max=1.0)

    return p


class Encoding(nn.Module): # 编码器
    def __init__(self, n_input_dims=3, device='cuda:0', direction:bool=False):
        super(Encoding, self).__init__()
        self.n_input_dims = n_input_dims
        self.device = device
        self.include_xyz = True
        self.direction = direction

        if not direction: # 如果不是方向
            encoding_config = {
                'otype': 'HashGrid',  # 'otype': 'Grid', 'type': 'Hash'
                'n_levels': 16,
                'n_features_per_level': 2,
                'log2_hashmap_size': 19,
                'base_resolution': 16,
                'per_level_scale': 1.447269237440378,  # "per_level_scale": 2.0,
                'include_xyz': self.include_xyz,
            }
            embed_dim = 3
        else:
            encoding_config = {
                'otype': 'SphericalHarmonics',
                'degree': 4,
            }
            embed_dim = 3

        with torch.cuda.device(device):
            encoding = tcnn.Encoding(n_input_dims=n_input_dims, encoding_config=encoding_config) # 编码

        self._B = nn.Parameter(torch.randn(n_input_dims, embed_dim) * torch.Tensor([25.0]))
        self.encoding = encoding
        self.n_output_dims = int(self.include_xyz) * embed_dim + self.encoding.n_output_dims

    def forward(self, x, *args):
        eps = 1e-5

        if self.direction:
            embedded_x = torch.sin(x @ self._B.to(x.device))
            # Expects 3D inputs that represent normalized vectors v transformed into the unit cube as (v+1)/2，表示归一化向量v的3D输入转换为单位立方体，如（v + 1）/ 2
            view_dirs = (embedded_x + 1) / 2
            assert view_dirs.min() >= 0-eps and view_dirs.max() <= 1+eps, f'dir value range ' \
                                                                  f'[{view_dirs.min().item(), view_dirs.max().item()}]!'
            out = self.encoding(view_dirs, *args)
        else:
            # assuming the x range within [-1, 1]
            embedded_x = x
            # Expects 3D inputs that represent normalized vectors v transformed into the unit cube as (v+1)/2.
            view_pts = (x + 1) / 2
            assert view_pts.min() >= 0-eps and view_pts.max() <= 1+eps, f'3d points value range ' \
                                                                          f'[{view_pts.min().item(), view_pts.max().item()}]!'

            out = self.encoding(view_pts, *args) # 编码

        if self.include_xyz:
            out = torch.cat([
                    embedded_x,
                    out,
                ], dim=-1)

        return out


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in: int = 3,
                 d_out: int = 32,
                 device: str ='cuda:0'):
        super(SDFNetwork, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device

        self.encoding = Encoding(n_input_dims=d_in, device=device, direction=False)
        self.sdf_layer = nn.Linear(self.encoding.n_output_dims, d_out)
        torch.nn.init.constant_(self.sdf_layer.bias, 0.0)
        torch.nn.init.constant_(self.sdf_layer.weight[:, 3:], 0.0)
        torch.nn.init.normal_(self.sdf_layer.weight[:, :3], mean=0.0, std=math.sqrt(2)/math.sqrt(d_out))

    def get_training_parameters(self, ignore_keys=()): # 获取训练参数
        params = {
            'network': list(self.sdf_layer.parameters()) + [self.encoding._B, ],
            'volume': list(self.encoding.encoding.parameters())
        }

        return params

    def forward(self, pts, bound=None):
        n_pts, _ = pts.shape

        if bound is not None:
            pts = normalized_3d_coordinate(pts, bound.to(pts.device))
        # pts should be range in [-1, 1]
        pts = self.encoding(pts) # 先进行哈希编码
        out = self.sdf_layer(pts)
        sdf, feat = out[:, 0:1], out[:, 1:]

        return sdf, feat

    def sdf(self, pts, bound=None, require_feature=False, require_gradient=False):
        if require_gradient:
            with torch.enable_grad():
                pts.requires_grad_(True) # [n_3dpoints, d_in]
                sdf, feat = self.forward(pts, bound) # 先进行哈希编码
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                # ! Distributed Data Parallel doesn't work with torch.autograd.grad()
                # ! (i.e., it will only work if gradients are to be accumulated in x.grad attributes of parameters)
                gradient = torch.autograd.grad(
                    outputs=sdf,
                    inputs=pts,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]  # [n_3dpoints, d_in]，优化sdf

            if require_feature:
                return sdf, feat, gradient
            else:
                return sdf, gradient
        else:
            sdf, feat = self.forward(pts, bound)
            if require_feature:
                return sdf, feat
            else:
                return sdf


class ColorNetwork(nn.Module):
    def __init__(self,
                 d_in: int = 3,
                 d_feat: int = 32 - 1,
                 d_hidden=64,
                 n_layers=2,
                 device='cuda:0'):
        super(ColorNetwork, self).__init__()
        self.d_in = d_in
        self.d_feat = d_feat
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.device = device

        embed_dim = 33
        self._B = nn.Parameter(torch.randn(3, embed_dim) * torch.Tensor([25.0]))

        # self.dir_encoding = Encoding(n_input_dims=d_in, device=self.device, direction=True)
        # n_input_dims = embed_dim + self.dir_encoding.n_output_dims + 1 + 3 + d_feat
        n_input_dims = embed_dim + 3 + d_feat

        with torch.cuda.device(device):
            network_config = {
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'none',
                'n_neurons': d_hidden,
                'n_hidden_layers': n_layers,
            }

            self.network = tcnn.Network(n_input_dims=n_input_dims, n_output_dims=3, network_config=network_config)

    def forward(self, view_pts, view_dirs, sdf, normals, feature_vectors):
        # view_dirs = self.dir_encoding(view_dirs)
        view_pts = torch.sin(view_pts @ self._B.to(view_pts.device))

        # refer to https://arxiv.org/abs/2003.09852
        rendering_input = torch.cat([view_pts, normals, feature_vectors], dim=1)

        x = self.network(rendering_input) #颜色网络

        x = torch.sigmoid(x)

        return x


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val=0.2, scale_factor=10.0):
        super(SingleVarianceNetwork, self).__init__()
        self.scale_factor = scale_factor
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        B, _ = x.shape
        return torch.ones(size=[B, 1]).to(x.device) * torch.exp(self.variance * self.scale_factor)


class InstantNeuS(nn.Module):
    def __init__(self, cfg, bound, device='cuda:0'):
        super(InstantNeuS, self).__init__()
        self.cfg = cfg
        self.register_buffer('bound', torch.tensor(bound).float())  # [3, 2]
        self.register_buffer('realtime_bound', torch.tensor(bound).float())  # [3, 2]
        self.device = device

        # SDFNetwork的输出维度d_out是32，这意味着它会输出一个32维的向量。在这32维中，第一维被用作SDF值，剩下的31维被用作特征向量。
        # ColorNetwork的输入维度d_feat是31，这是因为它接收来自SDFNetwork的31维特征向量作为输入，用于计算颜色。
        # 所以，ColorNetwork的d_feat比SDFNetwork的d_out小1，是因为SDFNetwork输出的第一维被用作SDF值，而不是特征向量，不会被传递给
        # ColorNetwork。
        self.sdf_network = SDFNetwork(**cfg['sdf_network'], device=device) #d_in=3, d_out=32,
        self.color_network = ColorNetwork(**cfg['color_network'], device=device) #d_in=3, d_feat=31, d_hidden=64, n_layers=2,
        self.variance_network = SingleVarianceNetwork(**cfg['variance_network']) #init_val=0.2, scale_factor=10.0
        self.sdf_smooth_std = cfg['sdf_smooth_std'] # 0.005
        self.sdf_sparse_factor = cfg['sdf_sparse_factor'] # 5
        self.sdf_truncation = cfg['sdf_truncation'] # 0.16
        self.sdf_random_weight = cfg['sdf_random_weight'] # 0.04
        self.cos_anneal_ratio = 1.0

    def get_training_parameters(self, ignore_keys=()): # 获取训练参数
        params = []
        all_params = {
            'sdf_network': list(self.sdf_network.get_training_parameters()['network']),
            'color_network': list(self.color_network.parameters()),
            'variance_network': list(self.variance_network.parameters()),
        }
        for k, v in all_params.items():
            if k not in ignore_keys:
                params += v

        return params

    def get_volume_parameters(self): # 获取体积参数
        params = list(self.sdf_network.get_training_parameters()['volume'])

        return params

    @torch.no_grad()
    def update_bound(self, bound): # 更新边界
        self.realtime_bound[:] = bound.float().to(self.realtime_bound.device)

    @torch.no_grad()
    def in_bound(self, pts, bound): # 判断点是否在边界内
        """
        Args:
            pts:                        (Tensor), 3d points
                                        [n_points, 3]
            bound:                      (Tensor), bound
                                        [3, 2]
        """
        # mask for points out of bound
        bound = bound.to(pts.device)
        mask_x = (pts[:, 0] < bound[0, 1]) & (pts[:, 0] > bound[0, 0])
        mask_y = (pts[:, 1] < bound[1, 1]) & (pts[:, 1] > bound[1, 0])
        mask_z = (pts[:, 2] < bound[2, 1]) & (pts[:, 2] > bound[2, 0])
        mask = (mask_x & mask_y & mask_z).bool()

        return mask

    def get_alpha(self, sdf, gradients, dirs, dists): # 获取alpha(输入sdf，gradient,射线方向，输出alpha)
        #alpha值用于确定每个体素对最终渲染图像的贡献。alpha值越大，体素对最终渲染图像的贡献越大。
        # 该方法接收四个参数：sdf（每个点的有符号距离函数值），gradients（每个点处的梯度向量），
        # dirs（从相机到每个点的方向向量），和dists（从相机到每个点的距离）。
        n_pts, _ = sdf.shape
        device = sdf.device
        # 首先，方法计算inv_s，这是SDF值的方差的倒数。这是通过将与sdf形状相同的全1张量传递给variance_network，
        # 然后剪辑结果以避免极大或极小值来完成的
        inv_s = self.variance_network(torch.ones([n_pts, 1]).to(device)).clip(1e-6, 1e6)
        # 接下来，方法计算true_cos，这是方向向量和梯度向量之间角度的余弦值。
        # 这是通过对dirs和gradients进行元素级乘法，然后沿第二维度求和来完成的。
        true_cos = (dirs * gradients).sum(dim=1, keepdim=True)  # v * n, [n_rays*n_samples, 1]
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        #“cos_anneal_ratio" 从开始的训练迭代中增长到1。下面的退火策略使cos值在开始的训练迭代中“不死”，以获得更好的收敛。

        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-negative
        # 然后，方法估计沿方向向量的下一个和前一个采样点处的SDF值，并计算这些点处的累积分布函数（CDF）值。
        # 然后，alpha值被计算为前一个和下一个CDF值之间的差，由前一个CDF值和一个小常数的和进行归一化，以避免除以零。
        est_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) / 2.0
        est_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) / 2.0
        prev_cdf = torch.sigmoid(est_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(est_next_sdf * inv_s)
        alpha = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clip(0.0, 1.0)

        return alpha

    def forward(self,
                rays_o,
                rays_d,
                z_vals,
                dists,
                render_params: dict = None):
        
        n_rays, n_samples = z_vals.shape # 获取采样射线数量和采样点数量
        device = z_vals.device

        z_vals = z_vals + dists / 2.0
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None] # 采样射线的采样点的3d坐标
        dirs = rays_d[:, None, :].expand(n_rays, n_samples, 3) # 射线方向
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        pts_mask = self.in_bound(pts, self.realtime_bound) # 判断点是否在边界内
        if torch.sum(pts_mask.float()) < 1:  # it may happen when render out image，当渲染出图像时可能会发生
            pts_mask[:100] = True

        def alpha_rgb_fn(pts_3d, dirs_3d, dists_3d, mask): # 计算SDF、RGB、alpha、梯度
            # [n_pts, 1], [n_pts, 3]
            n_pts, _ = pts_3d.shape
            out_sdf, out_feat, out_gradients = self.sdf_network.sdf(pts_3d[mask], self.bound,
                                                                    require_feature=True,
                                                                    require_gradient=True) #先算SDF
            sdf = torch.ones([n_pts, 1]).to(device) * 100
            feat = torch.zeros([n_pts, out_feat.shape[1]]).to(device).to(out_feat.dtype)
            gradients = torch.zeros([n_pts, 3]).to(device).to(out_gradients.dtype)

            sdf[mask] = out_sdf # mask是判断点是否在边界内的掩码
            feat[mask] = out_feat
            gradients[mask] = out_gradients

            alpha = self.get_alpha(sdf, gradients, dirs_3d, dists_3d)  # [n_pts, 1]
            out_rgb = self.color_network(pts_3d[mask], dirs_3d[mask], sdf[mask], gradients[mask], feat[mask])  # [n_pts, 3]

            rgb = torch.zeros([n_pts, 3]).to(device).to(out_rgb.dtype)
            rgb[mask] = out_rgb

            return sdf, rgb, alpha, gradients # 返回SDF、RGB、alpha、梯度

        sdf, rgb, alpha, gradients = alpha_rgb_fn(pts, dirs, dists, pts_mask) # 计算SDF、RGB、alpha、梯度
        sdf = sdf.reshape(n_rays, n_samples)
        rgb = rgb.reshape(n_rays, n_samples, 3)
        alpha = (alpha * pts_mask[:, None]).reshape(n_rays, n_samples)
        gradients = gradients.reshape(n_rays, n_samples, 3)
        pts_mask = pts_mask.reshape(n_rays, n_samples)

        weights = alpha * torch.cumprod(torch.cat([
            torch.ones([n_rays, 1]).to(device),
            1 - alpha + 1e-7,
        ], dim=1), dim=1)[:, :-1]  # [n_rays, n_samples]，torch.cumprod是累乘
        weight_sum = weights.sum(dim=1, keepdim=True)  # [n_rays, 1]
        rgb = (rgb * weights[:, :, None]).sum(dim=1, keepdim=False)

        depth = (z_vals * weights).sum(dim=1, keepdim=True)  # [n_rays, 1]，深度
        depth_vars = ((z_vals - depth) ** 2 * weights[:, :n_samples]).sum(dim=1, keepdim=True)  # [n_rays, 1]，深度的方差
        normals = gradients * weights[:, :, None]
        normals = (normals * pts_mask[:, :, None]).sum(dim=1, keepdim=False)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients, ord=2, dim=2, keepdim=False) - 1.0) ** 2  # [n_pts,]
        gradient_error = gradient_error * pts_mask
        gradient_error = gradient_error.mean().unsqueeze(dim=0)  # [1, ]

        return {
            'color': rgb,  # [n_rays, 3]
            'depth': depth,  # [n_rays, 1]
            'depth_variance': depth_vars,  # [n_rays, 1]
            'normal': normals,  # [n_rays, 3]
            'weight_sum': weight_sum,  # [n_rays, 1]
            'sdf_variance':  (1.0 / self.variance_network(torch.ones_like(depth))),  # [n_rays, 1]
            'sdf': sdf, # [n_rays, n_samples]
            'z_vals': z_vals,  # [n_pts, n_samples]
            'gradient_error': gradient_error,  # [1, ]
        }


    def compute_sdf_error(self, sdf, z_vals, gt_depth): # 计算sdf损失
        N_rays, N_surface = z_vals.shape

        pred_sdf = sdf.reshape(N_rays, N_surface)  # [n_rays, n_surface]
        truncation = self.sdf_truncation
        gt_depth = gt_depth.reshape(N_rays, 1)
        valid_mask = (gt_depth > 0).reshape(-1)
        gt_depth = gt_depth[valid_mask]
        z_vals = z_vals[valid_mask]
        pred_sdf = pred_sdf[valid_mask]

        front_mask = z_vals < (gt_depth - truncation)  # [n_rays, n_surface]
        bound = (gt_depth - z_vals)
        sdf_mask = bound.abs() <= truncation  # [n_rays, n_surface]

        n_valid_samples = front_mask.sum(dim=1, keepdim=False) + \
                          sdf_mask.sum(dim=1, keepdim=False) + 1e-8  # [n_rays,]
        n_valid_rays = valid_mask.sum()

        # refer to https://arxiv.org/pdf/2204.02296v2.pdf Eq(6)
        front_loss = (torch.max(
            torch.exp((-self.sdf_sparse_factor * pred_sdf).clamp(max=10.0)) - torch.ones_like(pred_sdf),
            pred_sdf - bound
        ).clamp(min=0.0)) * front_mask
        sdf_front_error = (front_loss.sum(dim=1, keepdim=False) / n_valid_samples).sum() / n_valid_rays
        sdf_error = torch.abs(pred_sdf - bound)
        sdf_error = ((sdf_error * sdf_mask).sum(dim=1, keepdim=False) / n_valid_samples).sum() / n_valid_rays

        return sdf_error, sdf_front_error


    @torch.no_grad()
    def extract_color(self, bound, vertices): # 提取颜色
        N = int(64 * 64 * 64)
        rgbs = []
        points = torch.from_numpy(vertices).float().to(bound.device)
        # 首先，函数将顶点转换为PyTorch张量，并将其分割成大小为N的块，
        # 然后，函数在每个块上进行迭代，并对每个块内的顶点调用self.sdf_network.sdf方法计算SDF值。
        # 这个方法需要点的坐标和场景的边界作为输入。
        for i, pts in enumerate(torch.split(points, N, dim=0)):
            sdf, feat, gradident = self.sdf_network.sdf(pts,
                                                        bound=bound,
                                                        require_feature=True,
                                                        require_gradient=True)
            # 接下来，函数调用self.color_network方法计算颜色。
            # 这个方法需要点的坐标、方向向量、SDF值、梯度向量和特征向量作为输入。
            out_rgb = self.color_network(pts, None, sdf, gradident, feat)
            rgbs.append(out_rgb.float()) #把sdf放入颜色网络渲染出rgb
        # 计算出的颜色被存储在一个数组rgbs中，这个数组的形状与顶点相同。
        rgbs = torch.cat(rgbs, dim=0)
        vertex_colors = rgbs.detach().cpu().numpy()
        vertex_colors = np.clip(vertex_colors, 0, 1) * 255
        vertex_colors = vertex_colors.astype(np.uint8)

        return vertex_colors

    # extract_fields函数的主要目的是提取3D空间中的场（field），这里特指的是有符号距离函数（Signed Distance
    # Function，简称SDF）的值。这个函数在体素网格中的每个点上计算SDF值，并将这些值存储在一个3D数组中。
    @torch.no_grad()
    def extract_fields(self, bound_min: torch.Tensor, bound_max: torch.Tensor, resolution: int): # 提取场
        N = 64
        # 首先，函数使用torch.linspace在每个维度上生成等间距的值，这些值的数量由resolution参数决定。
        # 这些值被分割成大小为N的块，其中N是一个预定义的常数。
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        bound = torch.cat([bound_min[:, None], bound_max[:, None]], dim=1)
        # 然后，函数在每个块上进行迭代，并使用torch.meshgrid在每个块内生成3D网格。
        # 这个网格代表了体素网格中的点。
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                        pts = torch.cat([
                            xx.reshape(-1, 1),
                            yy.reshape(-1, 1),
                            zz.reshape(-1, 1),
                        ], dim=1).to(bound.device)
                        mask = self.in_bound(pts, self.realtime_bound)
                        n_pts, _ = pts.shape
                        device = pts.device
                        sdf = torch.ones([n_pts, 1]).to(device) * 100
                        if mask.sum() > 0:
                            # 对于每个点，函数调用self.sdf_network.sdf方法计算SDF值。这个方法需要点的坐标和场景的边界作为输入。
                            val = self.sdf_network.sdf(pts[mask],
                                                       bound=bound,
                                                       require_feature=False,
                                                       require_gradient=False) #计算sdf
                            sdf[mask] = val
                        sdf = sdf.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N:xi * N + len(xs), yi * N:yi * N + len(ys), zi * N:zi * N + len(zs)] = -sdf

        return u
    # extract_geometry方法提取3D空间中的几何形状。用mcubes和trimesh软件包建立mesh
    @torch.no_grad()
    def extract_geometry(self,
                         resolution: int,
                         threshold: float,
                         c2w_ref: None,
                         save_path='./mesh.ply',
                         color=False):
        bound_min = self.bound[:, 0]
        bound_max = self.bound[:, 1]
        # 首先，函数调用extract_fields方法提取3D空间中的场（这里特指的是有符号距离函数(简称SDF）的值）。
        # 这个方法在体素网格中的每个点上计算SDF值，并将这些值存储在一个3D数组中。
        u = self.extract_fields(bound_min, bound_max, resolution)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()
        # 然后，函数使用mcubes.marching_cubes方法进行等值面提取，生成3D模型的顶点和三角形。
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        # vertices, triangles = go_mcubes.marching_cubes(-u, threshold, 3.0)
        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        # 接着，函数将生成的顶点从体素空间转换到世界空间。如果提供了c2w_ref参数，函数会使用这个变换矩阵进行转换。
        if c2w_ref is not None:
            c2w_ref = c2w_ref.cpu().numpy()  # [4, 4]
            vertices_homo = np.concatenate([vertices, np.ones_like(vertices[:, :1])], axis=1)
            # [1, 4, 4] @ [n_pts, 4, 1] = [n_pts, 4, 1]
            vertices = np.matmul(c2w_ref[None, :, :], vertices_homo[:, :, None])[:, :3, 0]

        vertex_colors = None
        # 如果color参数为True，函数会调用extract_color方法提取顶点颜色。
        if color:
            vertex_colors = self.extract_color(bound=self.bound.clone(), vertices=vertices)

        # 用trimesh软件包建立mesh
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)

        eps = 0.01
        bound = self.realtime_bound.detach().cpu().numpy()
        vertices = mesh.vertices[:, :3]
        bound_mask = np.all(vertices >= (bound[:, 0] - eps), axis=1) & np.all(vertices <= (bound[:, 1] + eps), axis=1)
        face_mask = bound_mask[mesh.faces].all(axis=1)
        mesh.update_faces(face_mask)
        mesh.remove_unreferenced_vertices()

        if save_path is not None:
            mesh.export(save_path)

        return mesh



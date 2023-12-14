import torch
import lietorch

from .geom import projective_ops as pops
from .modules.corr import CorrBlock


class MotionFilter:
    # 用于过滤输入的视频帧并提取特征，关键帧选择策略。
    """ This class is used to filter incoming frames and extract features """
    def __init__(self, net, video, thresh=2.5, device='cuda:0'):
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization，图像归一化的均值和标准差
        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    # __context_encoder方法接收一个图像并返回该图像的上下文特征。
    # 这些特征是通过将图像传递给cnet网络并将结果分割为两部分得到的
    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image): #内容编码网络
        # 特征提取
        """ context features """
        # image: [1, b, 3, h, w], net: [1, b, 128, h//8, w//8], inp: [1, b, 128, h//8, w//8]
        # 把三通道的图像分割成两部分，一部分是net，一部分是inp，然后分别经过tanh和relu激活函数，每个部分的通道数都是128
        net, inp = self.cnet(image).split([128, 128], dim=2)
        return net.tanh().squeeze(dim=0), inp.relu().squeeze(dim=0)


    # __feature_encoder处理的方式是将图像传递给fnet网络，然后将网络的输出沿着批处理维度（dim = 0）压缩，即删除批处理维度。这样，
    # 输出的张量的形状从[1, b, 128, h // 8, w // 8]
    # 变为[b, 128, h // 8, w // 8]，其中b是批处理大小，h和w是图像的高度和宽度，128
    # 是特征的通道数。  这个方法的主要目的是从输入图像中提取特征，这些特征随后可以用于计算相关性体积，这是视频帧的运动过滤和特征提取的一部分。
    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image): #特征编码网络，也算特征提取

        """ feature for correlation volume """
        # image: [1, b, 3, h, w], return: [1, b, 128, h//8, w//8]
        return self.fnet(image).squeeze(dim=0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, timestamp, image, depth=None, intrinsic=None, gt_pose=None):
        # 主要的更新操作，每帧视频都会运行
        """ main update operation - run on every frame in video """

        scale_factor = 8.0 # 缩放因子
        IdentityMat = lietorch.SE3.Identity(1, ).data.squeeze() # 单位矩阵

        batch, _, imh, imw = image.shape # batch: 1, imh: 图像高度, imw: 图像宽度
        ht = imh // scale_factor # ht: 缩放后的图像高度
        wd = imw // scale_factor # wd: 缩放后的图像宽度

        # normalize images, [b, 3, imh, imw] -> [1, b, 3, imh, imw], b=1 for mono, b=2 for stereo
        # 图像归一化
        inputs = image.unsqueeze(dim=0).to(self.device)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features，提取特征
        gmap = self.__feature_encoder(inputs) # [b, c, imh//8, imw//8]

        ### always add frist frame to the depth video ###
        # 总是将第一帧添加到深度视频中
        left_idx = 0 # i.e., left image, for stereo case, we only store the hidden or input of left image
        if self.video.counter.value == 0: # 第一帧
            net, inp = self.__context_encoder(inputs[:, [left_idx,]])  # [1, 128, imh//8, imw//8]，特征提取
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(timestamp, image[left_idx], IdentityMat, 1.0, depth,
                              intrinsic/scale_factor, gmap, net[left_idx], inp[left_idx], gt_pose) # 添加到视频（关键帧）中

        ### only add new frame if there is enough motion ###
        # 只有当有足够的运动（光流）时才添加新帧
        else:
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None, None]  # 生成一个坐标网格coords0,[1, 1, imh//8, imw//8, 2]
            corr = CorrBlock(self.fmap[None, [left_idx]], gmap[None, [left_idx]])(coords0)  #使用CorrBlock模块计算相关性体积corr, [1, 1, 4*49, imh//8, imw//8]

            # approximate flow magnitude using 1 update iteration, 放入Droid-slam进行1次更新迭代，近似计算光流大小
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)  # [1, 1, imh//8, imw//8, 2]

            # check motion magnitude / add new frame to video, 检查运动幅度/将新帧添加到视频中
            if delta.norm(dim=-1).mean().item() > self.thresh: # 如果光流大小大于阈值，则添加新关键帧
                self.count = 0
                net, inp = self.__context_encoder(inputs[:, [left_idx]])  # [1, 128, imh//8, imw//8]
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(timestamp, image[left_idx], None, None, depth,
                                  intrinsic/scale_factor, gmap, net[left_idx], inp[left_idx], gt_pose) # 添加新关键帧

            else:
                self.count += 1
import torch
import torch.nn.functional as F
from torch import nn

from . import layers_123821KB as layers


class BaseASPPNet(nn.Module):
    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations)

        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        h = self.aspp(h)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        self.stg1_low_band_net = BaseASPPNet(2, 32)
        self.stg1_high_band_net = BaseASPPNet(2, 32)

        self.stg2_bridge = layers.Conv2DBNActiv(34, 16, 1, 1, 0)
        self.stg2_full_band_net = BaseASPPNet(16, 32)

        self.stg3_bridge = layers.Conv2DBNActiv(66, 32, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(32, 64)

        self.out = nn.Conv2d(64, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux2_out = nn.Conv2d(32, 2, 1, bias=False)

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.offset = 128

    def forward(self, x, aggressiveness=None):

        # 将频谱拿出来，不参与梯度变化
        # 这么写是为了创建一个副本
        # 如果是 mix = x, 这两个值指向一个位置， x变则mix变
        mix = x.detach()

        # 新的x不会影响以前的x
        x = x.clone()

        # 由于对称性，取一半
        x = x[:, :, : self.max_bin]

        # 测一下一半频谱的大小
        bandw = x.size()[2] // 2

        # 高频低频分开处理，这里直接对半开
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),
                self.stg1_high_band_net(x[:, :, bandw:]),
            ],
            dim=2,
        )

        # 将高低频与原图谱进行拼接
        h = torch.cat([x, aux1], dim=1)

        #特征提取
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        # 再次拼接，再次提取特征
        h = torch.cat([x, aux1, aux2], dim=1)
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        # 融合之后，通过mask。一些被保留，一些被遮盖
        mask = torch.sigmoid(self.out(h))

        # 对mask进行填充，在第四个维度复制了边界的值
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        if self.training:
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode="replicate",
            )

            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode="replicate",
            )
            return mask * mix,   aux1 * mix,    aux2 * mix
        
        
        else:
            if aggressiveness:
                mask[:, :, : aggressiveness["split_bin"]] = torch.pow(
                    mask[:, :, : aggressiveness["split_bin"]],
                    1 + aggressiveness["value"] / 3,
                )
                mask[:, :, aggressiveness["split_bin"] :] = torch.pow(
                    mask[:, :, aggressiveness["split_bin"] :],
                    1 + aggressiveness["value"],
                )

            # 通过mask对图谱进行选择性更改，保留一部分，忽略一部分
            # 用来进行下一步的音源分离任务
            return mask * mix

    def predict(self, x_mag, aggressiveness=None):
        h = self.forward(x_mag, aggressiveness)

        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]
            assert h.size()[3] > 0

        return h

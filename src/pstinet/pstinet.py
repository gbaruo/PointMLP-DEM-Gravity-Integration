"""PSTINet：Physics-guided Spatial Terrain Inference Network（论文创新点②）。

动机
----
方域积分（zone_integration，Nagy 棱柱解析解）计算地形改正精度高，但对上亿格网点
逐一积分代价巨大。PSTINet 是一个轻量卷积网络：输入以目标点为中心的局部地形高程
patch（已减去中心点高程，表征相对地形起伏），直接输出该点的地形改正值（mGal）。

    输入  : (B, 1, H, W)  以目标点为中心的相对高程 patch（单位 m）
    输出  : (B, 1)        该中心点的地形改正值（单位 mGal）

训练采用「物理引导知识蒸馏」：以方域积分的精确解析解作为监督标签（teacher），
配合物理一致性正则（见 losses.py），使网络在保持物理意义的前提下大幅加速推理。

设计要点
--------
* 输入做「中心化」：patch 减去中心像元高程。地形改正只依赖相对高差，这样可去除
  绝对高程带来的偏置，增强跨区域泛化。
* 编码器用若干 Conv-BN-ReLU 下采样提取多尺度地形特征，全局池化后回归到标量。
* 训练/推理均要求 torch；torch 缺失时给出明确报错（此为可选高级功能）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover - 取决于环境
    _HAS_TORCH = False


@dataclass
class PSTINetConfig:
    """PSTINet 网络与输入配置（默认值与 config/terrain_correction.yaml 对应）。"""
    patch_size: int = 64             # 输入地形 patch 边长（像元）
    in_channels: int = 1             # 输入通道（相对高程，1）
    base_channels: int = 32          # 首层卷积通道数
    depth: int = 4                   # 下采样层数（决定感受野与参数量）
    dropout: float = 0.1             # 回归头 dropout
    center_normalize: bool = True    # 是否对 patch 做中心化（减中心像元高程）


if _HAS_TORCH:

    class ConvBlock(nn.Module):
        """Conv(3x3)-BN-ReLU x2 + 步长2下采样，提取并压缩地形特征。"""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

        def forward(self, x):
            x = self.body(x)
            x = self.down(x)
            return x

    class PSTINet(nn.Module):
        """地形改正回归网络：相对高程 patch -> 地形改正标量(mGal)。"""

        def __init__(self, config: Optional[PSTINetConfig] = None):
            super().__init__()
            self.cfg = config or PSTINetConfig()
            chans = [self.cfg.in_channels]
            c = self.cfg.base_channels
            for _ in range(self.cfg.depth):
                chans.append(c)
                c *= 2
            # 构建编码器
            blocks = []
            for i in range(self.cfg.depth):
                blocks.append(ConvBlock(chans[i], chans[i + 1]))
            self.encoder = nn.Sequential(*blocks)
            self.gap = nn.AdaptiveAvgPool2d(1)            # 全局平均池化 -> (B,C,1,1)
            feat_dim = chans[-1]
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, feat_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(feat_dim // 2, 1),              # 回归到地改值(mGal)
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """前向：x 形状 (B,1,H,W)，返回 (B,1) 地形改正值。"""
            if self.cfg.center_normalize:
                # 减去每个 patch 的中心像元高程，仅保留相对地形起伏
                b, _, h, w = x.shape
                center = x[:, :, h // 2, w // 2].view(b, 1, 1, 1)
                x = x - center
            feat = self.encoder(x)
            feat = self.gap(feat)
            out = self.head(feat)
            return out

    def build_model(config: Optional[PSTINetConfig] = None) -> "PSTINet":
        """工厂函数：按配置构建 PSTINet。"""
        return PSTINet(config or PSTINetConfig())

    def count_parameters(model: "nn.Module") -> int:
        """统计可训练参数量，便于在论文中报告模型规模。"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

else:  # torch 不可用时的占位，给出明确指引

    class PSTINet:  # type: ignore
        """占位类：未安装 PyTorch。安装后方可使用 PSTINet。"""

        def __init__(self, *_, **__):
            raise ImportError(
                "PSTINet 需要 PyTorch。请先安装：pip install torch。"
                "（神经推理为可选加速路径；不安装不影响方域积分主流程。）"
            )

    def build_model(*_, **__):
        raise ImportError("PSTINet 需要 PyTorch。请先安装：pip install torch。")

    def count_parameters(*_, **__):
        raise ImportError("PSTINet 需要 PyTorch。请先安装：pip install torch。")

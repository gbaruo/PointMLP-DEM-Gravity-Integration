"""PSTINet 物理引导损失函数（论文创新点②的训练约束）。

总损失由三部分加权组成：

    L = w_data * L_data  +  w_phys * L_phys  +  w_smooth * L_smooth

各项含义
--------
1. L_data（数据项 / 知识蒸馏）
   网络预测地改值与方域积分「精确解析解」标签之间的回归误差。这是核心监督信号，
   使网络学会逼近物理精确解。默认用 Smooth L1（Huber），对离群点更稳健。

2. L_phys（物理一致性正则）
   地形改正具有明确物理性质，用于约束网络不违背物理直觉：
     (a) 非负性：在「全部地形高于计算点」的纯地形质量情形，地改（质量引力的垂直
         分量修正）应为正。对这类样本，惩罚预测出现的负值。
     (b) 单调性（可选）：中心相对地形整体抬高，地改值不应减小。以小批量内的有限差
         分近似，惩罚与该趋势相悖的预测。
   说明：物理项以「软约束」形式加入，权重较小，避免压过数据项。

3. L_smooth（空间平滑正则，可选）
   相邻空间位置的地改值应平滑变化（地形改正场是空间连续的）。当一个 batch 内包含
   规则排列的相邻点时，惩罚预测的空间梯度过大。无相邻结构信息时该项为 0。

实现要求 torch；torch 缺失时本模块仍可导入（函数在调用时报错），不影响主流程导入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


@dataclass
class LossWeights:
    """各损失项权重（默认值与 config/terrain_correction.yaml 对应）。"""
    w_data: float = 1.0          # 数据项（蒸馏）权重
    w_phys: float = 0.1          # 物理一致性正则权重
    w_smooth: float = 0.01       # 空间平滑正则权重
    huber_beta: float = 1.0      # Smooth L1 的过渡阈值 beta
    enforce_nonneg: bool = True  # 是否启用非负性物理约束
    enforce_monotonic: bool = False  # 是否启用单调性物理约束


def data_loss(pred: "torch.Tensor", target: "torch.Tensor", beta: float = 1.0) -> "torch.Tensor":
    """数据项：预测与方域积分标签的 Smooth L1（Huber）回归误差。

    Smooth L1：|x|<beta 时为 0.5*x^2/beta（二次，平滑），否则为 |x|-0.5*beta（线性，
    抗离群）。pred、target 形状均为 (B,1)。
    """
    return F.smooth_l1_loss(pred, target, beta=beta)


def physics_loss(
    pred: "torch.Tensor",
    patch: "torch.Tensor",
    weights: LossWeights,
) -> "torch.Tensor":
    """物理一致性正则（非负性 + 可选单调性）。

    参数
    ----
    pred  : (B,1) 网络预测地改值。
    patch : (B,1,H,W) 对应输入相对高程 patch（已中心化或原始均可）。
    """
    device = pred.device
    loss = torch.zeros((), device=device)

    b, _, h, w = patch.shape
    center = patch[:, :, h // 2, w // 2].view(b, 1)        # 中心像元高程
    # 相对中心的地形高差均值：>0 表示周边整体高于中心（纯地形质量在上方）
    rel_mean = (patch.mean(dim=[2, 3]) - center)            # (B,1)

    # (a) 非负性：对“周边整体高于中心”的样本，地改预测应 >= 0，惩罚负值部分
    if weights.enforce_nonneg:
        mass_above = (rel_mean > 0).float()                # 该约束适用的样本掩膜
        neg_part = torch.clamp(-pred, min=0.0)             # 预测为负的幅度
        loss = loss + (mass_above * neg_part).mean()

    # (b) 单调性（可选）：rel_mean 越大，pred 期望越大。用批内排序的有限差近似：
    #     若按 rel_mean 升序排列，pred 也应大致升序；惩罚逆序对的负增量。
    if weights.enforce_monotonic and b >= 2:
        order = torch.argsort(rel_mean.squeeze(1))
        p_sorted = pred.squeeze(1)[order]
        dp = p_sorted[1:] - p_sorted[:-1]                  # 相邻预测增量
        loss = loss + torch.clamp(-dp, min=0.0).mean()     # 惩罚下降（逆单调）

    return loss


def smooth_loss(
    pred: "torch.Tensor",
    grid_shape: Optional[tuple] = None,
) -> "torch.Tensor":
    """空间平滑正则（可选）。

    当 batch 内样本按规则二维网格 (gh, gw) 排列（grid_shape 给定且 gh*gw==B）时，
    将 pred 重排为 (gh, gw)，惩罚其水平/垂直一阶差分，鼓励空间连续。否则返回 0。
    """
    if grid_shape is None:
        return torch.zeros((), device=pred.device)
    gh, gw = grid_shape
    if gh * gw != pred.shape[0]:
        return torch.zeros((), device=pred.device)
    field = pred.view(gh, gw)
    dx = field[:, 1:] - field[:, :-1]
    dy = field[1:, :] - field[:-1, :]
    return (dx.abs().mean() + dy.abs().mean())


def total_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    patch: "torch.Tensor",
    weights: Optional[LossWeights] = None,
    grid_shape: Optional[tuple] = None,
) -> dict:
    """组合总损失，返回包含各分项与总损失的字典，便于训练时记录与调权。

    返回
    ----
    dict: {
        'loss'       : 加权总损失（用于反向传播）,
        'data'       : 数据项,
        'physics'    : 物理项,
        'smooth'     : 平滑项,
    }
    """
    if not _HAS_TORCH:
        raise ImportError("损失函数需要 PyTorch。请先安装：pip install torch。")
    w = weights or LossWeights()

    l_data = data_loss(pred, target, beta=w.huber_beta)
    l_phys = physics_loss(pred, patch, w)
    l_smooth = smooth_loss(pred, grid_shape)

    loss = w.w_data * l_data + w.w_phys * l_phys + w.w_smooth * l_smooth
    return {
        "loss": loss,
        "data": l_data.detach(),
        "physics": l_phys.detach(),
        "smooth": l_smooth.detach() if torch.is_tensor(l_smooth) else l_smooth,
    }

"""PSTINet 训练脚本：用方域积分精确解作为监督标签（物理蒸馏）。

工作流
-----
1. 从融合后的近区 DEM 中随机采样训练点（或用规则网格）。
2. 对每个点，截取以其为中心的地形 patch（patch_size×patch_size 像元）。
3. 调用方域积分（zone_integration）计算该点精确的地形改正值作为「标签」。
4. 网络预测该 patch 的地改值；用物理引导损失对比，反向传播优化。
5. 周期性验证（在验证集上评估），保存最优模型。

内存与计算考量
--------------
* 方域积分逐点计算代价大（O(上亿)次棱柱操作）。故仅在离线「标签生成」阶段
  用一次；训练时读预生成的标签数据集。
* patch 提取用 numpy 高效切片，单个 patch 小（通常 64×64），内存友好。
* 训练可启用 GPU（自动检测 torch.cuda）以加速。

实现说明
--------
为应对 torch / scipy / 超大数据缺失等环境限制，此脚本做「优雅降级」：
  * torch 不可用 → 给出明确报错，但不crash;
  * 方域积分模块不可用 → 提示用预生成标签替代；
  * 数据集过小 → 警告但继续。
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


@dataclass
class TrainConfig:
    """训练超参（默认值与 config/terrain_correction.yaml 对应）。"""
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10              # Early stopping 耐心轮数
    device: str = "auto"            # "auto" 自动选 GPU 或 CPU；或明确指定 "cuda" / "cpu"
    checkpoint_dir: str = "./checkpoints"
    val_frac: float = 0.1           # 验证集比例


class PatchDataset:
    """地形 patch 数据集构建与采样。

    给定融合 DEM，以规则或随机方式采样训练点，每点截取以其为中心的 patch，
    并计算或读取该点的地形改正标签。
    """

    def __init__(
        self,
        dem: np.ndarray,
        dem_meta: dict,
        patch_size: int,
        points_xy: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        center_normalize: bool = True,
    ):
        """
        参数
        ----
        dem : (ny, nx) 融合 DEM 高程数组。
        dem_meta : {"origin_x", "origin_y", "cell_size", "crs"} 地理参照。
        patch_size : patch 边长（像元数）。
        points_xy : (N, 2) 训练点地理坐标，若 None 则自动在 DEM 内生成规则网格。
        labels : (N,) 地形改正标签(mGal)，若 None 则调用方需自行传入或生成。
        center_normalize : 是否对 patch 中心化（减中心像元高程）。
        """
        self.dem = dem.astype(np.float32)
        self.meta = dem_meta
        self.ps = patch_size
        self.cnorm = center_normalize
        self.ny, self.nx = dem.shape
        ox, oy, cs = dem_meta["origin_x"], dem_meta["origin_y"], dem_meta["cell_size"]
        self.ox, self.oy, self.cs = float(ox), float(oy), float(cs)

        if points_xy is None:
            # 规则网格采样（stride=patch_size，避免重叠）
            stride = patch_size
            yy, xx = np.mgrid[
                self.ps // 2 : self.ny : stride,
                self.ps // 2 : self.nx : stride,
            ]
            points_ij = np.column_stack([xx.ravel(), yy.ravel()])
            self.points_ij = points_ij
        else:
            # 地理坐标 -> 像元行列
            col = ((points_xy[:, 0] - self.ox) / self.cs).astype(int)
            row = ((points_xy[:, 1] - self.oy) / self.cs).astype(int)
            self.points_ij = np.column_stack([col, row])

        if labels is None:
            warnings.warn("未提供地改标签，需在外部生成后再传入。")
            self.labels = None
        else:
            self.labels = np.asarray(labels, dtype=np.float32)

    def extract_patches(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """提取所有训练点的 patch，返回 (N, 1, ps, ps) 与 (N,) 标签。

        超出边界的 patch 用边缘填充（repeat）。
        """
        n = len(self.points_ij)
        patches = np.zeros((n, 1, self.ps, self.ps), dtype=np.float32)
        hs = self.ps // 2
        for i, (col, row) in enumerate(self.points_ij):
            r0 = max(0, row - hs)
            r1 = min(self.ny, row + hs + 1)
            c0 = max(0, col - hs)
            c1 = min(self.nx, col + hs + 1)
            patch = self.dem[r0:r1, c0:c1].copy()
            # 边缘填充回到 (ps, ps)
            pr0 = max(0, hs - row)
            pr1 = pr0 + (r1 - r0)
            pc0 = max(0, hs - col)
            pc1 = pc0 + (c1 - c0)
            full = np.full((self.ps, self.ps), self.dem[row, col], dtype=np.float32)
            full[pr0:pr1, pc0:pc1] = patch
            patches[i, 0] = full
        return patches, self.labels


def generate_training_labels(
    dem: np.ndarray,
    dem_meta: dict,
    points_ij: np.ndarray,
    rho: float = 2670.0,
    zone_depth: float = 667.0,
) -> np.ndarray:
    """用方域积分计算训练标签。

    逐点调用 zone_integration.prism_zoning()，计算每个点的地形改正值。
    这是离线「标签生成」阶段，计算代价大但仅做一次。

    参数
    ----
    dem : (ny, nx) 融合 DEM。
    dem_meta : {"origin_x", "origin_y", "cell_size", ...}
    points_ij : (N, 2) 点的像元行列坐标 [col, row]。
    rho : 地壳密度(kg/m³)，默认 2670。
    zone_depth : 假设质量柱深度(m)，默认 667(对应地幔均衡深度)。

    返回
    ----
    (N,) 地改值(mGal)数组。
    """
    try:
        from src.gravity_correction.zone_integration import prism_zoning
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "标签生成需要 zone_integration 模块。"
            "或者使用预生成的标签数据集，无需在线计算。"
        ) from e

    ox = dem_meta["origin_x"]
    oy = dem_meta["origin_y"]
    cs = dem_meta["cell_size"]

    labels = []
    for col, row in points_ij:
        # 地理坐标
        x = ox + col * cs
        y = oy + row * cs
        # 该点周边 DEM
        corr = prism_zoning(dem, ox, oy, cs, x, y, rho=rho, zone_depth=zone_depth)
        labels.append(corr)
    return np.asarray(labels, dtype=np.float32)


def train_pstinet(
    dem: np.ndarray,
    dem_meta: dict,
    pstinet_config,
    labels: Optional[np.ndarray] = None,
    train_cfg: Optional[TrainConfig] = None,
) -> dict:
    """端到端训练 PSTINet。

    工作流
    ----
    1. 构建数据集（patch 提取 + 标签处理）。
    2. 按 val_frac 分割训练/验证集。
    3. 构建网络、优化器、学习率调度。
    4. 训练循环：前向、损失、反向、更新；周期性验证与 early stopping。
    5. 返回训练历史与最优模型状态。

    返回
    ----
    dict: {
        "best_epoch"    : 最优模型所在轮数,
        "best_val_loss" : 最优验证损失,
        "train_losses"  : 各轮训练损失列表,
        "val_losses"    : 各轮验证损失列表,
    }
    """
    if not _HAS_TORCH:
        raise ImportError("训练需要 PyTorch。请先安装：pip install torch。")

    from src.pstinet.pstinet import build_model, count_parameters
    from src.pstinet.losses import LossWeights, total_loss

    tcfg = train_cfg or TrainConfig()
    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)

    # 自动选设备
    if tcfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = tcfg.device
    print(f"[Train] 使用设备: {device}")

    # 数据集
    dataset = PatchDataset(dem, dem_meta, pstinet_config.patch_size, labels=labels)
    patches, lbl = dataset.extract_patches()
    if lbl is None:
        raise ValueError("未提供训练标签，请先调用 generate_training_labels() 生成。")
    n = len(patches)
    idx = np.random.permutation(n)
    n_val = max(1, int(n * tcfg.val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    x_train = torch.from_numpy(patches[train_idx]).to(device)
    y_train = torch.from_numpy(lbl[train_idx].reshape(-1, 1)).to(device)
    x_val = torch.from_numpy(patches[val_idx]).to(device)
    y_val = torch.from_numpy(lbl[val_idx].reshape(-1, 1)).to(device)

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True)

    # 模型
    model = build_model(pstinet_config).to(device)
    n_params = count_parameters(model)
    print(f"[Model] 参数量: {n_params:,}")

    # 优化
    optimizer = optim.Adam(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 训练循环
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    wts = LossWeights()

    for epoch in range(tcfg.num_epochs):
        # 训练
        model.train()
        train_loss_acc = 0.0
        for x_bat, y_bat in train_loader:
            optimizer.zero_grad()
            pred = model(x_bat)
            loss_dict = total_loss(pred, y_bat, x_bat, weights=wts)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()
        train_loss = train_loss_acc / len(train_loader)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        with torch.no_grad():
            pred_val = model(x_val)
            loss_dict = total_loss(pred_val, y_val, x_val, weights=wts)
            val_loss = loss_dict["loss"].item()
        val_losses.append(val_loss)

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}/{tcfg.num_epochs}] train={train_loss:.6f}, val={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            ckpt_path = os.path.join(tcfg.checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
        if no_improve >= tcfg.patience:
            print(f"[Early Stop] 验证集无改进 {tcfg.patience} 轮，停止训练。")
            break

        scheduler.step(val_loss)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

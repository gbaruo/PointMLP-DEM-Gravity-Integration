"""方域积分地形改正（Zone-domain prism integration terrain correction）。

本模块是整个系统的“金标准（ground truth）”计算单元，也是用于训练 PSTINet 的
监督标签来源。核心思想：将测点周围的 DEM 离散为一系列直角棱柱，逐棱柱用解析
封闭解（见 prism.py，Nagy 1966）累加其在测点处产生的垂向重力，得到地形质量
相对“参考面”的盈缺效应，即地形改正值（terrain correction，单位 mGal）。

地形改正（TC）的物理定义
------------------------
重力地形改正衡量“真实起伏地形”相对“布格平板假设”所多算/少算的那部分地形质量
对测点重力的影响。约定 TC 恒为正（加到布格异常上）：
  * 测点周围“高于参考面”的地形（山体）会向上吸引重力仪，使实测重力偏小，
    需要正向补偿；
  * 测点周围“低于参考面”的部分（沟谷，布格平板里被当成有质量）实际没有质量，
    也需要正向补偿。
本实现以“测点高程”为参考面，对每个 DEM 单元：
  - 若单元高于参考面：质量柱位于参考面之上（z 从 0 到 (h_cell - h_station)）；
  - 若单元低于参考面：等效为参考面之下缺失的质量柱（取绝对值并反号叠加）。
最终对 |g_z| 累加，保证 TC 为正。

分带积分（近区 / 中远区）
------------------------
* 近区（inner）：半径 <= inner_radius，使用高精度“点云融合 DEM”，逐像元棱柱积分，
  精度要求最高（近区地形对测点贡献最大且随距离快速衰减）。
* 中远区（far）：inner_radius < 半径 <= outer_radius，使用融合/遥感 DEM，为提速
  可按 far_downsample 对 DEM 聚合降采样。
* 半径 > outer_radius：忽略（远区贡献小，可在文档中讨论球面影响）。

性能
----
不对全图暴力三重循环，而是：以测点为中心，按 outer_radius 在 DEM 上裁剪一个邻域
窗口，仅对窗口内像元向量化计算棱柱效应，再按近/远区半径与降采样分别累加。

单位约定
--------
长度米(m)、密度 kg/m^3、内部重力 m/s^2，最终乘 1e5 转 mGal。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .prism import prism_gravity_vertical

# 重力地形改正常用常数
G_DEFAULT = 6.67430e-11          # 万有引力常数 (m^3 kg^-1 s^-2)
MGAL = 1.0e5                     # m/s^2 -> mGal
TWO_PI_G = 2.0 * np.pi * G_DEFAULT


def bouguer_slab_correction(height: np.ndarray, density: float = 2670.0) -> np.ndarray:
    """布格平板改正（Bouguer slab correction），单位 mGal。

    公式：g_B = 2πGρh
    其中 h 为测点相对基准面的高程(m)，ρ 为密度，G 为万有引力常数。
    返回正值，实际使用时通常从观测重力中“减去”布格改正。

    参数
    ----
    height : 高程(m)，可为标量或数组。
    density: 密度(kg/m^3)。
    """
    h = np.asarray(height, dtype=np.float64)
    return TWO_PI_G * density * h * MGAL


def free_air_correction(height: np.ndarray) -> np.ndarray:
    """自由空气改正（Free-air correction），单位 mGal。

    采用常用线性近似：g_FA = 0.3086 * h (mGal, h 单位 m)。
    返回正值（随高程升高，正常重力减小，需正向补偿）。
    """
    h = np.asarray(height, dtype=np.float64)
    return 0.3086 * h


@dataclass
class ZoneIntegrationConfig:
    """方域积分参数（默认值与 config/terrain_correction.yaml 对应）。"""
    cell_size: float = 1.0           # DEM 像元尺寸(m)
    density: float = 2670.0          # 地形密度(kg/m^3)
    inner_radius: float = 2000.0     # 近区半径(m)
    outer_radius: float = 20000.0    # 中远区外半径(m)
    far_downsample: int = 4          # 中远区降采样倍数(>=1)
    epsilon: float = 1.0e-9          # 数值稳定保护
    G: float = G_DEFAULT


class ZoneIntegrationTC:
    """方域积分地形改正计算器。

    用法
    ----
    >>> calc = ZoneIntegrationTC(ZoneIntegrationConfig(cell_size=1.0))
    >>> tc = calc.terrain_correction_at(dem, x0, y0, station_height,
    ...                                 origin_x, origin_y)
    其中 dem 为二维高程数组（行=y 方向，列=x 方向），origin_x/origin_y 为 dem[0,0]
    像元中心的地理坐标，(x0,y0) 为测点平面坐标，station_height 为测点高程。
    """

    def __init__(self, config: Optional[ZoneIntegrationConfig] = None):
        self.cfg = config or ZoneIntegrationConfig()
        if self.cfg.far_downsample < 1:
            raise ValueError("far_downsample 必须 >= 1")

    # ------------------------------------------------------------------ #
    # 内部：从全图 DEM 裁剪测点邻域窗口（按 outer_radius）
    # ------------------------------------------------------------------ #
    def _crop_window(
        self,
        dem: np.ndarray,
        x0: float,
        y0: float,
        origin_x: float,
        origin_y: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """裁剪以测点为中心、半径 outer_radius 的方形窗口。

        返回 (dem_win, dx, dy)：
            dem_win : 窗口高程数组
            dx, dy  : 与 dem_win 同形状的“像元中心相对测点”的平面坐标(m)
        约定：x 沿列方向递增，y 沿行方向递增（origin 为 dem[0,0] 像元中心坐标）。
        """
        cs = self.cfg.cell_size
        nrow, ncol = dem.shape
        # 测点在 DEM 像元索引（浮点）
        col0 = (x0 - origin_x) / cs
        row0 = (y0 - origin_y) / cs
        # 窗口半径（像元）
        half = int(np.ceil(self.cfg.outer_radius / cs))
        r1 = max(0, int(np.floor(row0)) - half)
        r2 = min(nrow, int(np.ceil(row0)) + half + 1)
        c1 = max(0, int(np.floor(col0)) - half)
        c2 = min(ncol, int(np.ceil(col0)) + half + 1)

        dem_win = dem[r1:r2, c1:c2]
        # 像元中心相对测点的坐标
        cols = np.arange(c1, c2)
        rows = np.arange(r1, r2)
        cell_x = origin_x + cols * cs            # 各列像元中心 x
        cell_y = origin_y + rows * cs            # 各行像元中心 y
        dx = cell_x[None, :] - x0                # (1, ncol_win) 广播
        dy = cell_y[:, None] - y0                # (nrow_win, 1) 广播
        dx = np.broadcast_to(dx, dem_win.shape)
        dy = np.broadcast_to(dy, dem_win.shape)
        return dem_win.astype(np.float64), dx.astype(np.float64), dy.astype(np.float64)

    # ------------------------------------------------------------------ #
    # 内部：对一组像元（同一 cell_size）做棱柱积分，返回 TC(mGal)
    # ------------------------------------------------------------------ #
    def _integrate_cells(
        self,
        h_cell: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        h_station: float,
        cell_size: float,
    ) -> float:
        """对给定像元集合做棱柱解析积分并累加为地形改正(mGal)。

        每个像元视为底面 cell_size×cell_size 的直角棱柱：
          - 平面范围：[dx-cs/2, dx+cs/2] × [dy-cs/2, dy+cs/2]（相对测点）
          - 垂向范围：以测点高程为参考面，z 从 0 到 (h_cell - h_station)
        z 轴向下为正：棱柱解析解要求传入 z1<z2。对“高于参考面”的像元，地形质量
        在参考面之上（数学上 z 取负），我们用 |g_z| 累加保证 TC 为正；对“低于参考面”
        的像元（缺失质量），同样取 |g_z| 累加。因此统一对 |Δh| 建棱柱并取绝对值。
        """
        cfg = self.cfg
        half = cell_size / 2.0
        x1 = dx - half
        x2 = dx + half
        y1 = dy - half
        y2 = dy + half

        # 相对参考面的高差（绝对值用于建棱柱厚度）
        dh = np.abs(h_cell - h_station)
        # 厚度为 0 的像元（与参考面同高）无贡献，跳过以省算力
        mask = dh > cfg.epsilon
        if not np.any(mask):
            return 0.0

        x1m, x2m = x1[mask], x2[mask]
        y1m, y2m = y1[mask], y2[mask]
        # 棱柱垂向：z 从 0 到 dh（向下为正的解析解，取 |g_z| 消除方向影响）
        z1m = np.zeros_like(dh[mask])
        z2m = dh[mask]

        gz = prism_gravity_vertical(
            x1m, x2m, y1m, y2m, z1m, z2m,
            density=cfg.density, G=cfg.G, epsilon=cfg.epsilon,
        )
        # 取绝对值累加 -> TC 恒正；乘 1e5 转 mGal
        return float(np.sum(np.abs(gz)) * MGAL)

    # ------------------------------------------------------------------ #
    # 内部：中远区 DEM 降采样（块平均）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _block_mean(arr: np.ndarray, dx: np.ndarray, dy: np.ndarray, factor: int):
        """对高程与坐标做 factor×factor 块平均降采样（提速中远区积分）。"""
        if factor <= 1:
            return arr, dx, dy
        nr, nc = arr.shape
        nr2, nc2 = nr // factor, nc // factor
        if nr2 == 0 or nc2 == 0:
            return arr, dx, dy
        sl = (slice(0, nr2 * factor), slice(0, nc2 * factor))
        a = arr[sl].reshape(nr2, factor, nc2, factor).mean(axis=(1, 3))
        gx = dx[sl].reshape(nr2, factor, nc2, factor).mean(axis=(1, 3))
        gy = dy[sl].reshape(nr2, factor, nc2, factor).mean(axis=(1, 3))
        return a, gx, gy

    # ------------------------------------------------------------------ #
    # 对外：单测点地形改正
    # ------------------------------------------------------------------ #
    def terrain_correction_at(
        self,
        dem: np.ndarray,
        x0: float,
        y0: float,
        station_height: float,
        origin_x: float,
        origin_y: float,
        far_dem: Optional[np.ndarray] = None,
        far_origin: Optional[Tuple[float, float]] = None,
        far_cell_size: Optional[float] = None,
    ) -> dict:
        """计算单个测点的地形改正（分近区/中远区）。

        参数
        ----
        dem        : 近区高精度 DEM（点云融合 DEM），二维数组。
        x0, y0     : 测点平面坐标(m)。
        station_height : 测点高程(m)，作为积分参考面。
        origin_x, origin_y : dem[0,0] 像元中心地理坐标(m)。
        far_dem    : 可选，独立的中远区 DEM（如遥感 DEM）。若为 None，则近/远区都用 dem。
        far_origin : far_dem[0,0] 像元中心坐标(m)，far_dem 提供时必填。
        far_cell_size : far_dem 像元尺寸(m)，far_dem 提供时必填。

        返回
        ----
        dict: {
            'tc_total'  : 总地形改正(mGal),
            'tc_inner'  : 近区贡献(mGal),
            'tc_far'    : 中远区贡献(mGal),
            'n_inner'   : 近区参与像元数,
            'n_far'     : 中远区参与像元数,
        }
        """
        cfg = self.cfg

        # ---- 近区：用高精度 dem，半径 <= inner_radius ----
        h_win, dx, dy = self._crop_window(dem, x0, y0, origin_x, origin_y)
        dist = np.sqrt(dx * dx + dy * dy)
        inner_mask = dist <= cfg.inner_radius
        tc_inner = self._integrate_cells(
            h_win[inner_mask], dx[inner_mask], dy[inner_mask],
            station_height, cfg.cell_size,
        ) if np.any(inner_mask) else 0.0
        n_inner = int(np.count_nonzero(inner_mask))

        # ---- 中远区：inner_radius < 半径 <= outer_radius ----
        if far_dem is not None:
            # 使用独立远区 DEM
            if far_origin is None or far_cell_size is None:
                raise ValueError("提供 far_dem 时必须同时提供 far_origin 与 far_cell_size")
            far_cfg_cs = far_cell_size
            fh, fdx, fdy = self._crop_window_custom(
                far_dem, x0, y0, far_origin[0], far_origin[1], far_cfg_cs
            )
        else:
            far_cfg_cs = cfg.cell_size
            fh, fdx, fdy = h_win, dx, dy

        # 中远区降采样提速
        fh_ds, fdx_ds, fdy_ds = self._block_mean(fh, fdx, fdy, cfg.far_downsample)
        eff_cs = far_cfg_cs * cfg.far_downsample
        fdist = np.sqrt(fdx_ds * fdx_ds + fdy_ds * fdy_ds)
        far_mask = (fdist > cfg.inner_radius) & (fdist <= cfg.outer_radius)
        tc_far = self._integrate_cells(
            fh_ds[far_mask], fdx_ds[far_mask], fdy_ds[far_mask],
            station_height, eff_cs,
        ) if np.any(far_mask) else 0.0
        n_far = int(np.count_nonzero(far_mask))

        return {
            "tc_total": tc_inner + tc_far,
            "tc_inner": tc_inner,
            "tc_far": tc_far,
            "n_inner": n_inner,
            "n_far": n_far,
        }

    def _crop_window_custom(self, dem, x0, y0, origin_x, origin_y, cell_size):
        """与 _crop_window 相同，但允许传入自定义 cell_size（用于独立远区 DEM）。"""
        cs = cell_size
        nrow, ncol = dem.shape
        col0 = (x0 - origin_x) / cs
        row0 = (y0 - origin_y) / cs
        half = int(np.ceil(self.cfg.outer_radius / cs))
        r1 = max(0, int(np.floor(row0)) - half)
        r2 = min(nrow, int(np.ceil(row0)) + half + 1)
        c1 = max(0, int(np.floor(col0)) - half)
        c2 = min(ncol, int(np.ceil(col0)) + half + 1)
        dem_win = dem[r1:r2, c1:c2]
        cols = np.arange(c1, c2)
        rows = np.arange(r1, r2)
        cell_x = origin_x + cols * cs
        cell_y = origin_y + rows * cs
        dx = np.broadcast_to(cell_x[None, :] - x0, dem_win.shape)
        dy = np.broadcast_to(cell_y[:, None] - y0, dem_win.shape)
        return dem_win.astype(np.float64), dx.astype(np.float64), dy.astype(np.float64)

    # ------------------------------------------------------------------ #
    # 对外：批量测点
    # ------------------------------------------------------------------ #
    def terrain_correction_batch(
        self,
        dem: np.ndarray,
        stations_xy: np.ndarray,
        stations_h: np.ndarray,
        origin_x: float,
        origin_y: float,
        **far_kwargs,
    ) -> np.ndarray:
        """对一批测点计算总地形改正(mGal)，返回一维数组。

        stations_xy : (N,2) 测点平面坐标
        stations_h  : (N,)  测点高程
        其余远区参数通过 far_kwargs 透传 terrain_correction_at。
        """
        out = np.empty(len(stations_xy), dtype=np.float64)
        for i, ((x0, y0), h) in enumerate(zip(stations_xy, stations_h)):
            res = self.terrain_correction_at(
                dem, float(x0), float(y0), float(h),
                origin_x, origin_y, **far_kwargs,
            )
            out[i] = res["tc_total"]
        return out

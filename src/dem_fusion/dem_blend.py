"""近区点云 DEM 与中远区遥感 DEM 的物理一致性融合（论文创新点①）。

问题背景
--------
* 近区 DEM 来自机载/地面点云：分辨率高、近区精度好，但覆盖范围有限。
* 中远区 DEM 来自遥感（SRTM/ASTER/COP30 等）：覆盖广，但分辨率与绝对精度较低，
  且与点云 DEM 之间常存在系统性高程基准差异（垂直基准、处理流程不同导致）。

若直接拼接两套 DEM 用于地形改正，会在拼接边界产生错台（高程跳变），导致地改值
出现非物理的突变。本模块通过两步实现物理一致的融合：

    第一步：高程基准对齐（datum alignment）
        在两套 DEM 的重叠区估计系统性高差，并消除之（中位数平移 / 最小二乘平面）。
    第二步：距离加权羽化融合（feathering blend）
        在过渡带内，用随距离平滑变化的权重把近区 DEM 平滑过渡到远区 DEM，
        消除拼接错台，公式： h = w*h_near + (1 - w)*h_far。

权重过渡函数
------------
设到近区有效边界的归一化距离为 t in [0,1]（0=边界内侧仍是纯近区，1=过渡带外侧
纯远区）。提供三种过渡：
    linear  ：w = 1 - t
    cosine  ：w = 0.5 * (1 + cos(pi * t))      （两端平滑、导数为 0，最自然）
    sigmoid ：w = 1 / (1 + exp(k * (t - 0.5))) （中间陡、两端缓）

单位与约定
--------
两套 DEM 须在同一平面坐标系；远区 DEM 会被重采样到近区 DEM 的统一格网后再融合。
高程、坐标、像元尺寸单位均为米(m)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class BlendConfig:
    """DEM 融合参数（默认值与 config/terrain_correction.yaml 对应）。"""
    transition_width: float = 200.0      # 过渡带宽度(m)
    method: str = "cosine"               # 过渡函数：linear | cosine | sigmoid
    sigmoid_k: float = 10.0              # sigmoid 陡度
    datum_align: str = "median"          # 基准对齐：none | median | plane
    resample: str = "bilinear"           # 远区重采样：nearest | bilinear | cubic
    nodata: float = -9999.0


def resample_to_grid(
    src: np.ndarray, src_ox: float, src_oy: float, src_cs: float,
    dst_ox: float, dst_oy: float, dst_cs: float, dst_shape: Tuple[int, int],
    method: str = "bilinear", nodata: float = -9999.0,
) -> np.ndarray:
    """将源 DEM 重采样到目标格网（与近区 DEM 对齐）。

    支持 nearest / bilinear / cubic（cubic 需 scipy；缺失时自动降级 bilinear）。
    目标格网 dst[0,0] 像元中心坐标为 (dst_ox, dst_oy)，行沿 y 递增、列沿 x 递增。
    """
    ny, nx = dst_shape
    dx = dst_ox + np.arange(nx) * dst_cs
    dy = dst_oy + np.arange(ny) * dst_cs
    gxx, gyy = np.meshgrid(dx, dy)

    col = (gxx - src_ox) / src_cs
    row = (gyy - src_oy) / src_cs

    if method == "nearest":
        out = _sample_nearest(src, row, col, nodata)
    elif method == "cubic":
        out = _sample_scipy(src, row, col, order=3, nodata=nodata)
    else:
        out = _sample_bilinear(src, row, col, nodata)
    return out


def _sample_nearest(src, row, col, nodata):
    """最近邻采样。"""
    r = np.round(row).astype(int)
    c = np.round(col).astype(int)
    valid = (r >= 0) & (r < src.shape[0]) & (c >= 0) & (c < src.shape[1])
    out = np.full(row.shape, nodata, dtype=np.float64)
    out[valid] = src[r[valid], c[valid]]
    return out


def _sample_bilinear(src, row, col, nodata):
    """双线性插值采样（纯 numpy）。"""
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r1, c1 = r0 + 1, c0 + 1
    valid = (r0 >= 0) & (r1 < src.shape[0]) & (c0 >= 0) & (c1 < src.shape[1])
    out = np.full(row.shape, nodata, dtype=np.float64)
    rr = row - r0
    cc = col - c0
    r0v, c0v = r0[valid], c0[valid]
    r1v, c1v = r1[valid], c1[valid]
    wa = (1 - rr[valid]) * (1 - cc[valid])
    wb = (1 - rr[valid]) * cc[valid]
    wc = rr[valid] * (1 - cc[valid])
    wd = rr[valid] * cc[valid]
    out[valid] = (
        wa * src[r0v, c0v] + wb * src[r0v, c1v]
        + wc * src[r1v, c0v] + wd * src[r1v, c1v]
    )
    return out


def _sample_scipy(src, row, col, order, nodata):
    """高阶插值采样（scipy.ndimage.map_coordinates）。缺失则降级双线性。"""
    try:
        from scipy.ndimage import map_coordinates
    except Exception:
        return _sample_bilinear(src, row, col, nodata)
    coords = np.vstack([row.ravel(), col.ravel()])
    sampled = map_coordinates(src, coords, order=order, mode="nearest")
    out = sampled.reshape(row.shape)
    invalid = (row < 0) | (row > src.shape[0] - 1) | (col < 0) | (col > src.shape[1] - 1)
    out[invalid] = nodata
    return out


def estimate_datum_offset(
    near: np.ndarray, far: np.ndarray, valid_mask: np.ndarray,
    mode: str = "median",
) -> Tuple[np.ndarray, dict]:
    """估计并消除远区 DEM 相对近区 DEM 的系统性高差，返回校正后的 far 与诊断信息。

    mode='median'：用重叠区高差中位数做整体平移（稳健，抗异常值）。
    mode='plane' ：用最小二乘拟合高差的一次平面 a*x+b*y+c，消除倾斜性系统差。
    mode='none'  ：不做对齐。
    valid_mask   ：两套 DEM 在该像元都有效（非 nodata）的布尔掩膜。
    """
    diff = near - far
    info = {"mode": mode}
    if mode == "none" or valid_mask.sum() < 10:
        info["offset"] = 0.0
        return far.copy(), info

    if mode == "median":
        off = float(np.median(diff[valid_mask]))
        info["offset"] = off
        return far + off, info

    if mode == "plane":
        ny, nx = near.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        X = xx[valid_mask].ravel().astype(np.float64)
        Y = yy[valid_mask].ravel().astype(np.float64)
        D = diff[valid_mask].ravel().astype(np.float64)
        A = np.column_stack([X, Y, np.ones_like(X)])
        coef, *_ = np.linalg.lstsq(A, D, rcond=None)
        a, b, c = coef
        plane = a * xx + b * yy + c
        info.update({"plane_coef": (float(a), float(b), float(c))})
        return far + plane, info

    raise ValueError(f"未知 datum_align 模式: {mode}")


def _weight_from_distance(t: np.ndarray, method: str, k: float) -> np.ndarray:
    """由归一化距离 t in [0,1] 计算近区权重 w in [0,1]（见模块 docstring 公式）。"""
    t = np.clip(t, 0.0, 1.0)
    if method == "linear":
        return 1.0 - t
    if method == "sigmoid":
        return 1.0 / (1.0 + np.exp(k * (t - 0.5)))
    return 0.5 * (1.0 + np.cos(np.pi * t))


@dataclass
class FusionResult:
    """融合结果容器。"""
    dem: np.ndarray
    weight: np.ndarray
    origin_x: float
    origin_y: float
    cell_size: float
    crs: str
    nodata: float
    datum_info: dict


def blend_dems(
    near_dem: np.ndarray, near_ox: float, near_oy: float, near_cs: float,
    far_dem: np.ndarray, far_ox: float, far_oy: float, far_cs: float,
    crs: str = "EPSG:4547",
    config: Optional[BlendConfig] = None,
) -> FusionResult:
    """融合近区点云 DEM 与中远区遥感 DEM。

    步骤
    ----
    1. 把远区 DEM 重采样到近区 DEM 的统一格网。
    2. 在重叠区（两者都有效）估计并消除系统性高差（基准对齐）。
    3. 计算近区权重图：以近区有效区域为核心，向外 transition_width 形成过渡带，
       过渡带内 w 由 1 平滑降到 0。
    4. h = w*h_near + (1-w)*h_far，得到无缝融合 DEM。
    """
    cfg = config or BlendConfig()
    ny, nx = near_dem.shape

    far_rs = resample_to_grid(
        far_dem, far_ox, far_oy, far_cs,
        near_ox, near_oy, near_cs, (ny, nx),
        method=cfg.resample, nodata=cfg.nodata,
    )

    near_valid = near_dem != cfg.nodata
    far_valid = far_rs != cfg.nodata
    both_valid = near_valid & far_valid

    far_aligned, datum_info = estimate_datum_offset(
        near_dem, far_rs, both_valid, mode=cfg.datum_align
    )
    far_aligned[~far_valid] = cfg.nodata

    weight = _compute_weight_map(near_valid, near_cs, cfg)

    out = np.full((ny, nx), cfg.nodata, dtype=np.float64)
    only_near = near_valid & (~far_valid)
    only_far = (~near_valid) & far_valid
    both = near_valid & far_valid

    out[only_near] = near_dem[only_near]
    out[only_far] = far_aligned[only_far]
    w = weight[both]
    out[both] = w * near_dem[both] + (1.0 - w) * far_aligned[both]

    return FusionResult(
        dem=out, weight=weight,
        origin_x=near_ox, origin_y=near_oy, cell_size=near_cs,
        crs=crs, nodata=cfg.nodata, datum_info=datum_info,
    )


def _compute_weight_map(near_valid: np.ndarray, cell_size: float, cfg: BlendConfig) -> np.ndarray:
    """计算近区权重图。

    思路：以近区有效像元为 1、无效为 0，计算每个像元到近区无效区的距离（即到近区
    边界的内向距离）。在边界向内 transition_width 范围内，权重由 0->1 平滑上升；
    更内部为纯近区（w=1）；近区之外为纯远区（w=0）。

    用 scipy 距离变换；若无 scipy，则退化为近区内 w=1、外 w=0 的硬边界。
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return near_valid.astype(np.float64)

    if cfg.transition_width <= 0:
        return near_valid.astype(np.float64)

    dist_in = distance_transform_edt(near_valid) * cell_size
    t = 1.0 - np.clip(dist_in / cfg.transition_width, 0.0, 1.0)
    w = _weight_from_distance(t, cfg.method, cfg.sigmoid_k)
    w[~near_valid] = 0.0
    return w

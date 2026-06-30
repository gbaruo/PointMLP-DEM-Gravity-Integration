"""由地面点生成规则格网高精度 DEM（近区 DEM）。

输入为地面分类后的地面点 (N,3)，输出规则格网 DEM（二维高程数组）及其地理参照
信息（origin、像元尺寸、CRS）。提供两种插值：

    method = 'idw'  ：反距离加权插值（Inverse Distance Weighting），稳健、处处可算。
    method = 'tin'  ：基于 Delaunay 三角网的线性插值（scipy.griddata linear），
                      在数据凸包内精度高、保持地形特征；凸包外用最近邻回填。

可选 Kriging（克里金）精度更高但计算成本大，这里标注 TODO，留作后续扩展。

输出可选写出 GeoTIFF（需 rasterio；未安装时仅返回 numpy 数组，不报错）。

单位与约定
--------
* 平面坐标与像元尺寸单位为米(m)，高程单位为米(m)。
* origin_x / origin_y 为 dem[0,0] 像元中心的地理坐标。
* 行索引沿 y 方向递增、列索引沿 x 方向递增（与 zone_integration 模块约定一致）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _lazy_import_rasterio():
    """惰性导入 rasterio（写 GeoTIFF 用）。缺失返回 None，由调用方降级。"""
    try:
        import rasterio  # noqa: F401
        from rasterio.transform import from_origin  # noqa: F401
        return rasterio
    except Exception:
        return None


@dataclass
class DEMGridConfig:
    """近区 DEM 生成参数（默认值与 config/terrain_correction.yaml 对应）。"""
    resolution: float = 1.0          # 近区 DEM 分辨率(m)
    method: str = "idw"              # 插值方法：'idw' 或 'tin'
    idw_power: float = 2.0           # IDW 距离幂次（越大越局部）
    idw_k: int = 12                  # IDW 每个格点参与的最近邻点数
    idw_search_radius: float = 0.0   # IDW 搜索半径(m)，0 表示仅用 k 近邻不限半径
    crs: str = "EPSG:4547"           # 坐标参考系（默认 CGCS2000 / 3-degree Gauss-Kruger）
    nodata: float = -9999.0          # 无数据填充值


@dataclass
class DEMResult:
    """DEM 生成结果容器。"""
    dem: np.ndarray                  # (ny, nx) 高程数组
    origin_x: float                  # dem[0,0] 像元中心 x
    origin_y: float                  # dem[0,0] 像元中心 y
    cell_size: float                 # 像元尺寸(m)
    crs: str                         # 坐标系
    nodata: float                    # 无数据值


def _build_grid_axes(
    points: np.ndarray, res: float
) -> Tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """根据点云范围与分辨率构建格网坐标轴。

    返回 (gx, gy, origin_x, origin_y, nx, ny)：
        gx, gy   : 各列/行像元中心坐标（1D）
        origin_* : dem[0,0] 像元中心坐标
        nx, ny   : 列数、行数
    """
    x, y = points[:, 0], points[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    nx = max(1, int(np.ceil((xmax - xmin) / res)) + 1)
    ny = max(1, int(np.ceil((ymax - ymin) / res)) + 1)
    gx = xmin + np.arange(nx) * res            # 列方向像元中心 x
    gy = ymin + np.arange(ny) * res            # 行方向像元中心 y
    return gx, gy, float(xmin), float(ymin), nx, ny


def dem_idw(points: np.ndarray, cfg: DEMGridConfig) -> DEMResult:
    """反距离加权（IDW）插值生成 DEM。

    公式：对每个格点 P，取其最近的 k 个地面点（可选限定搜索半径），
        z(P) = Σ w_i z_i / Σ w_i,   w_i = 1 / d_i^p
    其中 d_i 为格点到第 i 个邻点的水平距离，p 为距离幂次（idw_power）。
    当某格点与某数据点重合（d=0）时直接取该点高程，避免除零。

    实现用 scipy.cKDTree 做快速最近邻查询，向量化计算所有格点。
    """
    from scipy.spatial import cKDTree

    res = cfg.resolution
    gx, gy, ox, oy, nx, ny = _build_grid_axes(points, res)

    # 生成所有格点坐标 (ny*nx, 2)
    gxx, gyy = np.meshgrid(gx, gy)             # 形状 (ny, nx)
    query_pts = np.column_stack([gxx.ravel(), gyy.ravel()])

    tree = cKDTree(points[:, :2])
    k = min(cfg.idw_k, len(points))
    # 查询每个格点的 k 近邻；distance_upper_bound 用于限定搜索半径
    if cfg.idw_search_radius and cfg.idw_search_radius > 0:
        dist, idx = tree.query(
            query_pts, k=k, distance_upper_bound=cfg.idw_search_radius
        )
    else:
        dist, idx = tree.query(query_pts, k=k)

    # k=1 时 scipy 返回一维，统一成二维方便处理
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    z = points[:, 2]
    # 超出搜索半径的邻点 idx==len(points)，标记为无效
    invalid = idx >= len(points)
    safe_idx = np.where(invalid, 0, idx)
    neigh_z = z[safe_idx]                       # (Npix, k)

    # 距离权重 w = 1/d^p；d=0（重合点）用极小值保护后会被下方“精确命中”覆盖
    with np.errstate(divide="ignore"):
        w = 1.0 / np.power(dist, cfg.idw_power)
    w[invalid] = 0.0                            # 无效邻点权重置 0
    w[np.isinf(w)] = 0.0                        # 先把 inf（d=0）置0，稍后单独处理

    # 处理“格点与数据点几乎重合”的情况：直接取该点高程
    exact_hit = np.isclose(dist, 0.0) & (~invalid)
    has_exact = exact_hit.any(axis=1)

    wsum = w.sum(axis=1)
    # 正常加权平均
    with np.errstate(invalid="ignore", divide="ignore"):
        zhat = (w * neigh_z).sum(axis=1) / wsum
    # 对有精确命中的格点，用命中点高程覆盖
    if has_exact.any():
        first_exact = np.argmax(exact_hit, axis=1)   # 每行第一个命中列
        zhat_exact = neigh_z[np.arange(len(neigh_z)), first_exact]
        zhat[has_exact] = zhat_exact[has_exact]
    # 完全没有有效邻点的格点（全部超半径）置 nodata
    no_neighbor = (wsum == 0) & (~has_exact)
    zhat[no_neighbor] = cfg.nodata

    dem = zhat.reshape(ny, nx)
    return DEMResult(dem, ox, oy, res, cfg.crs, cfg.nodata)


def dem_tin(points: np.ndarray, cfg: DEMGridConfig) -> DEMResult:
    """基于 Delaunay 三角网的线性插值生成 DEM（scipy.griddata 'linear'）。

    在地面点的凸包内：用三角面线性插值，保形、精度高；
    凸包外：griddata linear 会产生 NaN，这里用最近邻（'nearest'）回填，避免空洞。
    """
    from scipy.interpolate import griddata

    res = cfg.resolution
    gx, gy, ox, oy, nx, ny = _build_grid_axes(points, res)
    gxx, gyy = np.meshgrid(gx, gy)

    pts2d = points[:, :2]
    z = points[:, 2]

    # 线性插值（凸包外为 NaN）
    zi = griddata(pts2d, z, (gxx, gyy), method="linear")
    # 凸包外用最近邻回填
    mask_nan = np.isnan(zi)
    if mask_nan.any():
        zi_near = griddata(pts2d, z, (gxx, gyy), method="nearest")
        zi[mask_nan] = zi_near[mask_nan]

    return DEMResult(zi, ox, oy, res, cfg.crs, cfg.nodata)


def generate_dem(points: np.ndarray, config: Optional[DEMGridConfig] = None) -> DEMResult:
    """统一入口：按 config.method 选择 IDW / TIN 生成近区 DEM。"""
    cfg = config or DEMGridConfig()
    method = (cfg.method or "idw").lower()
    if method == "idw":
        return dem_idw(points, cfg)
    if method == "tin":
        return dem_tin(points, cfg)
    if method == "kriging":
        # TODO: 克里金插值（变差函数建模），精度高但成本大，后续实现；暂回退 IDW。
        print("[generate_dem] kriging 暂未实现，回退 IDW。")
        return dem_idw(points, cfg)
    raise ValueError(f"未知 DEM 插值方法: {cfg.method}")


def write_geotiff(result: DEMResult, path: str) -> bool:
    """将 DEM 写出为 GeoTIFF。成功返回 True；无 rasterio 则跳过并返回 False。

    注意 GeoTIFF 行序通常自上而下（北在上）。本项目内部约定行索引沿 y 递增
    （南在上）；写出时做上下翻转，使 GeoTIFF 北在上，并据此构造 transform。
    """
    rasterio = _lazy_import_rasterio()
    if rasterio is None:
        print("[write_geotiff] 未安装 rasterio，跳过 GeoTIFF 写出（仅返回数组）。")
        return False
    from rasterio.transform import from_origin

    dem = result.dem
    ny, nx = dem.shape
    cs = result.cell_size
    # 内部 origin 是 dem[0,0]（最小 y）像元中心；翻转后顶行对应最大 y
    top_y = result.origin_y + (ny - 1) * cs
    # from_origin 取左上角坐标（像元左上角），故用中心减半个像元
    west = result.origin_x - cs / 2.0
    north = top_y + cs / 2.0
    transform = from_origin(west, north, cs, cs)

    dem_north_up = np.flipud(dem)              # 翻转为北在上
    with rasterio.open(
        path, "w", driver="GTiff", height=ny, width=nx, count=1,
        dtype="float32", crs=result.crs, transform=transform,
        nodata=result.nodata,
    ) as dst:
        dst.write(dem_north_up.astype("float32"), 1)
    return True

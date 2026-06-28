"""统一的输出接口：地改结果导出为 GeoTIFF / NPY / 可视化图像 + 元数据。

支持格式
--------
* GeoTIFF (.tif)  : 地理参考光栅，支持 GIS 直接读取；需 rasterio。
* NPY (.npy)      : NumPy 二进制格式，快速读写；不含地理信息，需配合 .json 元数据。
* PNG (.png)      : 伪彩色可视化（地改值映射到冷-热色表），便于快速浏览；需 matplotlib。
* JSON (.json)    : 地理参照与处理参数元数据（CRS、origin、分辨率、生成时间等）。

工作流
-----
用户调用 export_result()，传入地改网格 + 元数据 + 输出路径，自动导出上述格式组合。
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


def _lazy_import_rasterio():
    """惰性导入 rasterio（GeoTIFF 写出用）。"""
    try:
        import rasterio
        from rasterio.transform import from_origin
        return rasterio
    except Exception:
        return None


def _lazy_import_matplotlib():
    """惰性导入 matplotlib（可视化用）。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        return plt, mcolors
    except Exception:
        return None, None


def write_geotiff(
    data: np.ndarray,
    path: str,
    origin_x: float,
    origin_y: float,
    cell_size: float,
    crs: str = "EPSG:4547",
    nodata: float = -9999.0,
    description: str = "",
) -> bool:
    """将地改网格写出为 GeoTIFF（带地理参照）。

    参数
    ----
    data : (ny, nx) 地改值网格(mGal)。
    path : 输出文件路径(.tif)。
    origin_x, origin_y : 像元 [0,0] 中心地理坐标。
    cell_size : 像元尺寸(m)。
    crs : 坐标参考系(默认 CGCS2000)。
    nodata : 无数据值。
    description : TIFF 标签(可选)。

    返回 True 若成功；False 若 rasterio 不可用。
    """
    rasterio = _lazy_import_rasterio()
    if rasterio is None:
        warnings.warn("rasterio 未安装，跳过 GeoTIFF 写出。pip install rasterio")
        return False

    from rasterio.transform import from_origin

    ny, nx = data.shape
    # GeoTIFF 行序北在上；内部约定南在上，故翻转
    data_north_up = np.flipud(data)
    top_y = origin_y + (ny - 1) * cell_size
    west = origin_x - cell_size / 2.0
    north = top_y + cell_size / 2.0
    transform = from_origin(west, north, cell_size, cell_size)

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=ny, width=nx, count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data_north_up.astype("float32"), 1)
        if description:
            dst.update_tags(1, description=description)
    return True


def write_npy(data: np.ndarray, path: str) -> bool:
    """将地改网格写出为 NPY（NumPy 二进制格式，快速高效）。

    返回 True 若成功。
    """
    try:
        np.save(path, data.astype("float32"))
        return True
    except Exception as e:
        warnings.warn(f"NPY 写出失败: {e}")
        return False


def write_visualization(
    data: np.ndarray,
    path: str,
    nodata: float = -9999.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdYlBu_r",
) -> bool:
    """将地改值可视化为伪彩色 PNG（便于快速浏览与审核）。

    参数
    ----
    data : (ny, nx) 地改值网格。
    path : 输出 PNG 路径。
    nodata : 无数据值（不参与色彩映射）。
    vmin, vmax : 颜色映射范围，None 自动由数据 min/max 确定。
    cmap : matplotlib 色表名称。

    返回 True 若成功；False 若 matplotlib 不可用。
    """
    plt, mcolors = _lazy_import_matplotlib()
    if plt is None:
        warnings.warn("matplotlib 未安装，跳过可视化。pip install matplotlib")
        return False

    data_viz = np.ma.masked_where(data == nodata, data)
    if vmin is None:
        vmin = np.nanpercentile(data[data != nodata], 2) if np.any(data != nodata) else 0
    if vmax is None:
        vmax = np.nanpercentile(data[data != nodata], 98) if np.any(data != nodata) else 100

    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    im = ax.imshow(data_viz, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    cbar = plt.colorbar(im, ax=ax, label="地形改正值 (mGal)")
    ax.set_title("地形改正值分布（Terrain Correction）", fontsize=14, fontweight="bold")
    ax.set_xlabel("列 (Column)")
    ax.set_ylabel("行 (Row)")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return True


def write_metadata(
    path: str,
    origin_x: float,
    origin_y: float,
    cell_size: float,
    crs: str,
    nodata: float,
    shape: tuple,
    description: str = "",
    processing_info: Optional[dict] = None,
) -> bool:
    """将处理参数与元数据写出为 JSON。

    便于后续读取时恢复地理参照与处理链路信息。

    参数
    ----
    path : 输出 JSON 路径。
    origin_x, origin_y : 像元 [0,0] 中心坐标。
    cell_size : 像元尺寸。
    crs : 坐标系。
    nodata : 无数据值。
    shape : (ny, nx) 网格形状。
    description : 处理描述。
    processing_info : 处理链路与参数(可选)。
    """
    meta = {
        "timestamp": datetime.now().isoformat(),
        "grid": {
            "origin_x": float(origin_x),
            "origin_y": float(origin_y),
            "cell_size": float(cell_size),
            "shape": list(shape),
            "crs": crs,
            "nodata": float(nodata),
        },
        "description": description,
        "processing": processing_info or {},
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        warnings.warn(f"JSON 写出失败: {e}")
        return False


def export_result(
    data: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell_size: float,
    output_dir: str,
    basename: str = "terrain_correction",
    crs: str = "EPSG:4547",
    nodata: float = -9999.0,
    formats: Optional[list] = None,
    description: str = "",
    processing_info: Optional[dict] = None,
) -> dict:
    """端到端导出地改结果：GeoTIFF + NPY + PNG + JSON 元数据。

    参数
    ----
    data : (ny, nx) 地改值网格(mGal)。
    origin_x, origin_y : 像元 [0,0] 中心坐标。
    cell_size : 像元尺寸(m)。
    output_dir : 输出目录。
    basename : 文件名前缀（不含扩展名）。
    crs : 坐标系。
    nodata : 无数据值。
    formats : 导出格式列表，默认 ["geotiff", "npy", "png", "json"]。
    description : 处理说明。
    processing_info : 处理链路/参数字典。

    返回
    ----
    dict: {
        "geotiff": 文件路径或 None,
        "npy": 文件路径或 None,
        "png": 文件路径或 None,
        "json": 文件路径或 None,
    }
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if formats is None:
        formats = ["geotiff", "npy", "png", "json"]
    formats = [f.lower() for f in formats]

    results = {"geotiff": None, "npy": None, "png": None, "json": None}
    ny, nx = data.shape

    # ---- GeoTIFF ----
    if "geotiff" in formats or "tif" in formats:
        tif_path = os.path.join(output_dir, f"{basename}.tif")
        if write_geotiff(data, tif_path, origin_x, origin_y, cell_size, crs, nodata, description):
            results["geotiff"] = tif_path
            print(f"✅ GeoTIFF 已写出: {tif_path}")
        else:
            print(f"⚠️  GeoTIFF 写出失败或 rasterio 不可用")

    # ---- NPY ----
    if "npy" in formats:
        npy_path = os.path.join(output_dir, f"{basename}.npy")
        if write_npy(data, npy_path):
            results["npy"] = npy_path
            print(f"✅ NPY 已写出: {npy_path}")

    # ---- PNG 可视化 ----
    if "png" in formats:
        png_path = os.path.join(output_dir, f"{basename}_viz.png")
        if write_visualization(data, png_path, nodata=nodata):
            results["png"] = png_path
            print(f"✅ PNG 可视化已写出: {png_path}")
        else:
            print(f"⚠️  PNG 可视化失败或 matplotlib 不可用")

    # ---- JSON 元数据 ----
    if "json" in formats:
        json_path = os.path.join(output_dir, f"{basename}_meta.json")
        if write_metadata(
            json_path, origin_x, origin_y, cell_size, crs, nodata, (ny, nx),
            description=description, processing_info=processing_info
        ):
            results["json"] = json_path
            print(f"✅ 元数据已写出: {json_path}")

    print(f"\n📁 输出文件都在: {output_dir}\n")
    return results

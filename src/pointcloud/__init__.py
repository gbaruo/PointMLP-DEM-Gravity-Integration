"""点云处理子包：地面分类 + 高精度近区 DEM 生成 + 上亿点云分块流式处理。

模块
----
    ground_filter        : 多算法地面分类（CSF 默认 / PMF / SMRF / deep）
    dem_from_pointcloud  : 地面点 -> 规则格网高精度 DEM（IDW / TIN）
    tiling               : 上亿级点云分瓦片流式处理（避免一次性载入内存）

设计原则
--------
* 重型可选依赖（laspy 读 las/laz）采用“惰性导入 + 优雅降级”，缺失时仍可处理
  .xyz / .npy 点云，不会导致整个包 import 失败。
"""

from .ground_filter import GroundFilter, ground_filter_csf

__all__ = ["GroundFilter", "ground_filter_csf"]

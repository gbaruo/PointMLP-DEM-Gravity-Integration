"""PointMLP-DEM-Gravity-Integration 主包。

核心模块
-------
* gravity_correction    : 方域积分(Nagy 棱柱解析解 + 方域分割)
* pointcloud            : 点云处理(地面分类 + DEM 生成 + 分瓦片流式)
* dem_fusion            : DEM 融合(高程基准对齐 + 羽化融合)
* pstinet               : 神经推理(物理引导知识蒸馏加速)
* method_select         : 方法推荐(根据数据规模自动选路径)
* output                : 统一输出(GeoTIFF/NPY/PNG + 元数据)
* terrain_correction    : 主入口(一行代码端到端)

快速开始
-------
    from src import TerrainCorrector
    
    # 自动配置 + 处理 + 导出
    tc = TerrainCorrector.auto_configure(n_points=5_000_000, precision="cm")
    tc.process_and_export(points_file="data.las", output_dir="./results")

论文创新点映射
-----------
    创新点①  : dem_fusion.blend_dems()         (近区点云DEM与中远区遥感DEM物理一致融合)
    创新点②  : pstinet (网络+损失+训练)        (物理引导神经推理快速路径)
    方法贡献 : terrain_correction.TerrainCorrector  (端到端自动化工作流)
"""

from .terrain_correction import TerrainCorrector, ProcessingState

__all__ = [
    "TerrainCorrector",
    "ProcessingState",
]

__version__ = "0.1.0"
__author__ = "gbaruo"
__description__ = "PointMLP-DEM-Gravity Integration: Physics-guided terrain correction"

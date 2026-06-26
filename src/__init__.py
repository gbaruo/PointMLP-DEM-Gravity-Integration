"""基于点云/遥感DEM融合的方域积分+PSTINet双路径自适应地形改正系统。

顶层包。各子模块：
    pointcloud         : 点云地面分类 + 高精度DEM生成
    dem_fusion         : 近区点云DEM 与 中远区遥感DEM 融合
    gravity_correction : 方域积分 / PSTINet / 地形判别 / 方法推荐 / 报表
    integration        : 端到端流程编排
"""

__version__ = "1.0.0"

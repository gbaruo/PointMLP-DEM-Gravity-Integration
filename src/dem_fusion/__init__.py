"""DEM 融合子包。

主路径（论文创新点①）
--------------------
    dem_blend : 近区点云 DEM 与中远区遥感 DEM 的物理一致性融合
                （高程基准对齐 + 距离加权羽化融合）。

说明
----
本子包中若存在 pimsr_model.py（DEM+SAR 超分占位实现），保留以兼容，但非主路径；
端到端流程使用 dem_blend 进行融合。
"""

from .dem_blend import (
    BlendConfig,
    FusionResult,
    blend_dems,
    resample_to_grid,
    estimate_datum_offset,
)

__all__ = [
    "BlendConfig",
    "FusionResult",
    "blend_dems",
    "resample_to_grid",
    "estimate_datum_offset",
]

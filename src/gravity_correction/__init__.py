"""重力地形改正子包。

导出主要类，方便外部以 `from src.gravity_correction import XXX` 调用。
注意：导入采用“尽量惰性”策略，重型依赖（如 torch）在真正使用时才报错，
避免仅使用解析方域积分时被迫安装深度学习库。
"""

from .prism import prism_gravity_vertical  # 棱柱解析解(纯 numpy，无重依赖)
from .zone_integration import (
    ZoneIntegrationTC,
    bouguer_slab_correction,
    free_air_correction,
)

__all__ = [
    "prism_gravity_vertical",
    "ZoneIntegrationTC",
    "bouguer_slab_correction",
    "free_air_correction",
]

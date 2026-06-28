"""方法推荐与选择引擎(根据用户数据规模、精度需求、计算资源自动匹配最优路径)。

工作流决策树
-----------
用户输入：数据量(点数), 精度需求(mm/cm/dm), 计算资源(GPU可用?)
系统输出：推荐的完整处理链路与参数配置

典型场景
--------
1. 小规模精确 (<100M点, mm级)
   路径: 点云分类 -> DEM近区 -> 方域积分全栅格 -> 地改网格
   适用: 关键点/验证点的高精度校正

2. 中等规模快速 (100M~1B点, cm级)  
   路径: 点云分类 -> DEM近区 -> DEM融合 -> 神经网络快速预测 -> 可选关键点方域积分校验
   适用: 大区域快速制图

3. 超大规模(>1B点, dm级)
   路径: 上亿点分瓦片流式 -> 瓦片DEM -> 融合后直用神经网络 -> 并行推理输出
   适用: 洲际/全球规模项目

参数推荐表
---------
精度需求   | 地面分类    | DEM分辨率  | 近区宽度  | 融合过渡带  | 推理方法
---------|-----------|----------|---------|-----------|--------
mm (高)    | CSF严格   | 0.5m    | 200m   | 100m      | 方域(全) + 神经(补)
cm (中)    | CSF默认   | 1m      | 500m   | 200m      | 神经(快) + 方域(验)
dm (低)    | PMF快速   | 2m      | 1000m  | 500m      | 神经(仅)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PrecisionLevel(Enum):
    """精度等级枚举。"""
    MM = "mm"      # 毫米级（cm以内）
    CM = "cm"      # 厘米级（dm以内）
    DM = "dm"      # 分米级（m以内）


class ComputeCapability(Enum):
    """计算能力枚举。"""
    CPU_ONLY = "cpu_only"
    GPU_AVAILABLE = "gpu_available"
    HPC = "hpc"


@dataclass
class MethodRecommendation:
    """方法推荐结果容器。"""
    scenario: str                      # 场景描述(如"小规模精确")
    chain: list                        # 推荐处理链路[module_name, ...]
    point_count_range: tuple           # 适用点数范围 (min, max)，None表无上限
    expected_time: str                 # 预期运行时间估计(如"2h on 32-core CPU")
    config_presets: dict               # 各模块配置的推荐值


def estimate_input_size(
    point_source: str,
    approx_n_points: Optional[int] = None,
    dem_file_path: Optional[str] = None,
) -> int:
    """估计输入数据量(点数)。

    参数
    ----
    point_source : "las" | "xyz" | "npy" | "dem" 输入类型
    approx_n_points : 用户提供的近似点数，优先使用
    dem_file_path : 若点源为"dem"，可用像元总数估算
    """
    if approx_n_points is not None:
        return int(approx_n_points)
    if point_source == "dem" and dem_file_path:
        try:
            import numpy as np
            dem = np.load(dem_file_path) if dem_file_path.endswith(".npy") else None
            if dem is not None:
                return int(dem.size)
        except Exception:
            pass
    # 默认估计
    return 1_000_000  # 100万点作为默认


def recommend_method(
    n_points: int,
    precision: PrecisionLevel = PrecisionLevel.CM,
    compute: ComputeCapability = ComputeCapability.CPU_ONLY,
) -> MethodRecommendation:
    """根据数据规模、精度需求、计算资源推荐最优处理路径。

    返回 MethodRecommendation，包含场景、链路、参数推荐等。
    """
    # ---- 小规模精确(<100M) ----
    if n_points < 100_000_000 and precision == PrecisionLevel.MM:
        return MethodRecommendation(
            scenario="小规模精确(mm级)",
            chain=[
                "pointcloud.ground_filter(csf_strict)",
                "pointcloud.dem_from_pointcloud(idw_k=16)",
                "dem_fusion.blend_dems(transition=100m)",
                "gravity_correction.prism_zoning(full_grid)",
            ],
            point_count_range=(1_000_000, 100_000_000),
            expected_time="0.5-2h on 8-core CPU",
            config_presets={
                "ground_filter": {"method": "csf", "csf.rigidness": 3, "csf.class_threshold": 0.3},
                "dem_from_pointcloud": {"method": "idw", "idw_k": 16, "resolution": 0.5},
                "dem_fusion": {"transition_width": 100.0, "datum_align": "plane"},
                "prism_zoning": {"zone_depth": 667.0},
            },
        )

    # ---- 中等规模快速(100M~1B, cm级) ----
    if 100_000_000 <= n_points < 1_000_000_000 and precision == PrecisionLevel.CM:
        method = "neural" if compute != ComputeCapability.CPU_ONLY else "zone"
        chain = [
            "pointcloud.ground_filter(csf)",
            "pointcloud.dem_from_pointcloud(idw)",
            "dem_fusion.blend_dems(transition=200m)",
        ]
        if compute != ComputeCapability.CPU_ONLY:
            chain.append("pstinet.inference(batch)")
        else:
            chain.append("gravity_correction.prism_zoning(sparse)")
        return MethodRecommendation(
            scenario=f"中等规模快速(cm级, {method})",
            chain=chain,
            point_count_range=(100_000_000, 1_000_000_000),
            expected_time="2-8h on 32-core CPU / 0.5-2h on GPU",
            config_presets={
                "ground_filter": {"method": "csf", "csf.rigidness": 2},
                "dem_from_pointcloud": {"method": "idw", "idw_k": 12, "resolution": 1.0},
                "dem_fusion": {"transition_width": 200.0, "datum_align": "median"},
                "pstinet": {"patch_size": 64, "batch_size": 32} if compute != ComputeCapability.CPU_ONLY else None,
            },
        )

    # ---- 超大规模(>1B, dm级或快速) ----
    if n_points >= 1_000_000_000:
        return MethodRecommendation(
            scenario="超大规模流式(dm级或快速)",
            chain=[
                "pointcloud.tiling(tile_size=500m)",
                "pointcloud.ground_filter(pmf_per_tile)",
                "pointcloud.dem_from_pointcloud(tin_per_tile)",
                "dem_fusion.blend_dems(mosaic)",
                "pstinet.inference(parallel_tiles)",
            ],
            point_count_range=(1_000_000_000, None),
            expected_time="4-24h on HPC cluster / 8-48h on 32-core CPU",
            config_presets={
                "tiling": {"tile_size": 500.0, "buffer": 50.0, "max_points_in_memory": 5_000_000, "n_workers": 4},
                "ground_filter": {"method": "pmf"},
                "dem_from_pointcloud": {"method": "tin", "resolution": 2.0},
                "dem_fusion": {"transition_width": 500.0},
                "pstinet": {"batch_size": 64, "n_workers": 4} if compute == ComputeCapability.HPC else {"batch_size": 32},
            },
        )

    # ---- 默认：中等规模cm级 ----
    return MethodRecommendation(
        scenario="中等规模(默认cm级)",
        chain=[
            "pointcloud.ground_filter(csf)",
            "pointcloud.dem_from_pointcloud(idw)",
            "dem_fusion.blend_dems()",
            "gravity_correction.zone_integration()",
        ],
        point_count_range=(10_000_000, 100_000_000),
        expected_time="1-4h on 8-core CPU",
        config_presets={
            "ground_filter": {"method": "csf"},
            "dem_from_pointcloud": {"method": "idw", "resolution": 1.0},
            "dem_fusion": {"transition_width": 200.0},
        },
    )


def print_recommendation(rec: MethodRecommendation) -> None:
    """打印友好的推荐说明。"""
    print("\n" + "="*70)
    print(f"📋 方法推荐：{rec.scenario}")
    print("="*70)
    print(f"✅ 推荐链路:")
    for i, step in enumerate(rec.chain, 1):
        print(f"   {i}. {step}")
    print(f"\n📊 适用数据范围: {rec.point_count_range[0]:,} ~ {rec.point_count_range[1] or '∞'} 点")
    print(f"⏱️  预期运行时间: {rec.expected_time}")
    print(f"\n⚙️  参数推荐:")
    for module, params in rec.config_presets.items():
        if params:
            print(f"   {module}:")
            for k, v in params.items():
                print(f"      {k}: {v}")
    print("="*70 + "\n")

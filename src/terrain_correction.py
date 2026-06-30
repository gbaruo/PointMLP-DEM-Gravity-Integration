"""主入口 TerrainCorrector：一句话从点云到地改网格的统一 API。

用法示例（论文演示代码）
----------------------
    from src.terrain_correction import TerrainCorrector
    
    # 方式 1：自动推荐（仅需指定数据量与精度）
    tc = TerrainCorrector.auto_configure(
        n_points=5_000_000,
        precision="cm",
        dem_file="merged_dem.tif"
    )
    tc.process_and_export(
        points_file="points.las",
        output_dir="./results"
    )
    
    # 方式 2：手工配置（细粒度控制）
    tc = TerrainCorrector(config_file="config/terrain_correction.yaml")
    tc.process(points_file="points.las")
    dem_corr, meta = tc.get_correction_grid()
    tc.export(dem_corr, meta, output_dir="./results")

设计原则
-------
* 零配置快速启动：auto_configure() 自动选参数。
* 配置文件驱动：YAML 配置所有子模块参数，便于复现与批处理。
* 渐进式处理：逐步返回中间结果（地面点、DEM、融合结果、地改网格）。
* 容错降级：模块缺失(torch/scipy等)时自动选替代路径,不中断。
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class ProcessingState:
    """处理状态容器：记录各阶段输出中间结果。"""
    points: Optional[np.ndarray] = None       # 原始点云
    ground_points: Optional[np.ndarray] = None  # 地面点
    dem_near: Optional[np.ndarray] = None     # 近区 DEM
    dem_far: Optional[np.ndarray] = None      # 远区 DEM
    dem_fusion: Optional[np.ndarray] = None   # 融合 DEM
    correction_grid: Optional[np.ndarray] = None  # 地改网格
    metadata: dict = None                     # 地理参照与处理参数


class TerrainCorrector:
    """地形改正端到端处理器：从原始数据到最终地改网格的统一入口。"""

    def __init__(self, config_file: Optional[str] = None):
        """
        参数
        ----
        config_file : YAML 配置文件路径。若 None，使用内置默认配置。
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.state = ProcessingState()

    def _load_config(self) -> dict:
        """加载 YAML 配置文件，或返回内置默认配置。"""
        if self.config_file and os.path.exists(self.config_file):
            try:
                import yaml
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                warnings.warn(f"配置文件读取失败({e})，使用默认配置")
        # 内置默认配置
        return {
            "ground_filter": {"method": "csf", "csf.rigidness": 2},
            "dem_from_pointcloud": {"method": "idw", "resolution": 1.0},
            "dem_fusion": {"transition_width": 200.0},
            "zone_integration": {"zone_depth": 667.0},
            "output": {"formats": ["geotiff", "npy", "json"]},
        }

    @classmethod
    def auto_configure(
        cls,
        n_points: int,
        precision: str = "cm",
        dem_file: Optional[str] = None,
    ) -> TerrainCorrector:
        """根据数据规模与精度需求自动配置。

        参数
        ----
        n_points : 点云点数。
        precision : 精度等级 "mm" | "cm" | "dm"。
        dem_file : 远区 DEM 文件路径(可选)。

        返回自动配置后的 TerrainCorrector 实例。
        """
        from src.method_select import recommend_method, PrecisionLevel

        prec_map = {"mm": PrecisionLevel.MM, "cm": PrecisionLevel.CM, "dm": PrecisionLevel.DM}
        prec = prec_map.get(precision, PrecisionLevel.CM)
        rec = recommend_method(n_points, prec)

        tc = cls(config_file=None)
        # 将推荐配置合并到 tc.config
        tc.config.update(rec.config_presets or {})
        tc.state.metadata = {"recommendation": rec.scenario, "dem_file": dem_file}
        return tc

    def process(self, points_file: str, dem_far_file: Optional[str] = None) -> ProcessingState:
        """端到端处理：从点云文件开始的完整流程。

        步骤
        ----
        1. 读点云
        2. 地面分类
        3. 建近区 DEM
        4. (若有远区 DEM) 融合
        5. 计算地改网格

        返回处理状态（含各阶段中间结果）。
        """
        from src.pointcloud.ground_filter import GroundFilter, GroundFilterConfig
        from src.pointcloud.dem_from_pointcloud import generate_dem, DEMGridConfig
        from src.dem_fusion.dem_blend import blend_dems, BlendConfig
        from src.gravity_correction.zone_integration import compute_zone_correction

        # ---- 步骤1：读点云 ----
        print("[Step 1] 读取点云...")
        try:
            from src.pointcloud.ground_filter import load_points
            points = load_points(points_file)
        except Exception as e:
            raise RuntimeError(f"点云读取失败: {e}")
        self.state.points = points
        print(f"  ✓ 读取 {len(points):,} 个点")

        # ---- 步骤2：地面分类 ----
        print("[Step 2] 地面分类...")
        gf_cfg = self.config.get("ground_filter", {})
        gf = GroundFilter(GroundFilterConfig(**gf_cfg))
        ground = gf.extract_ground(points)
        self.state.ground_points = ground
        print(f"  ✓ 分类出 {len(ground):,} 个地面点 ({100*len(ground)/len(points):.1f}%)")

        # ---- 步骤3：建近区 DEM ----
        print("[Step 3] 建近区 DEM...")
        dem_cfg = self.config.get("dem_from_pointcloud", {})
        dem_near_res = generate_dem(ground, DEMGridConfig(**dem_cfg))
        self.state.dem_near = dem_near_res.dem
        self.state.metadata = {
            "near_origin_x": dem_near_res.origin_x,
            "near_origin_y": dem_near_res.origin_y,
            "cell_size": dem_near_res.cell_size,
            "crs": dem_near_res.crs,
        }
        print(f"  ✓ 建成 {dem_near_res.dem.shape} 近区 DEM")

        # ---- 步骤4：融合(若有远区 DEM) ----
        dem_fusion = dem_near_res
        if dem_far_file and os.path.exists(dem_far_file):
            print("[Step 4] DEM 融合...")
            try:
                import numpy as np
                dem_far = np.load(dem_far_file) if dem_far_file.endswith(".npy") else None
                if dem_far is not None:
                    blend_cfg = self.config.get("dem_fusion", {})
                    blend_res = blend_dems(
                        dem_near_res.dem, dem_near_res.origin_x, dem_near_res.origin_y, dem_near_res.cell_size,
                        dem_far, 0.0, 0.0, 1.0,  # 远区 DEM 坐标假设(需实际提供)
                        config=BlendConfig(**blend_cfg),
                    )
                    self.state.dem_fusion = blend_res.dem
                    self.state.metadata["blend_info"] = blend_res.datum_info
                    print(f"  ✓ 融合完成 (基准对齐: {blend_res.datum_info.get('mode', 'N/A')})")
            except Exception as e:
                warnings.warn(f"DEM 融合失败({e})，仅使用近区 DEM")
        else:
            self.state.dem_fusion = dem_near_res.dem

        # ---- 步骤5：计算地改网格 ----
        print("[Step 5] 计算地形改正...")
        dem_for_corr = self.state.dem_fusion or self.state.dem_near
        zi_cfg = self.config.get("zone_integration", {})
        try:
            corr_grid = compute_zone_correction(
                dem_for_corr,
                dem_near_res.origin_x, dem_near_res.origin_y, dem_near_res.cell_size,
                **zi_cfg
            )
            self.state.correction_grid = corr_grid
            print(f"  ✓ 地改网格计算完成: 范围 {corr_grid.min():.1f}~{corr_grid.max():.1f} mGal")
        except Exception as e:
            warnings.warn(f"地改计算失败: {e}")

        return self.state

    def get_correction_grid(self) -> Tuple[np.ndarray, dict]:
        """获取地改网格与元数据。"""
        if self.state.correction_grid is None:
            raise ValueError("尚未处理，请先调用 process()。")
        return self.state.correction_grid, self.state.metadata

    def export(
        self,
        correction_grid: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
        output_dir: str = "./results",
        basename: str = "terrain_correction",
    ) -> dict:
        """导出结果到多种格式。"""
        from src.output import export_result

        grid = correction_grid or self.state.correction_grid
        meta = metadata or self.state.metadata or {}

        if grid is None:
            raise ValueError("无地改网格可导出，请先调用 process()。")

        origin_x = meta.get("near_origin_x", 0.0)
        origin_y = meta.get("near_origin_y", 0.0)
        cell_size = meta.get("cell_size", 1.0)
        crs = meta.get("crs", "EPSG:4547")

        formats = self.config.get("output", {}).get("formats", ["geotiff", "npy", "json"])
        print(f"\n[导出] 写出到 {output_dir}...")
        results = export_result(
            grid, origin_x, origin_y, cell_size, output_dir, basename,
            crs=crs, formats=formats, processing_info=meta
        )
        return results

    def process_and_export(
        self,
        points_file: str,
        dem_far_file: Optional[str] = None,
        output_dir: str = "./results",
    ) -> dict:
        """一行代码：处理 + 导出（演示用）。"""
        print(f"\n{'='*70}")
        print(f"🚀 地形改正端到端处理开始")
        print(f"{'='*70}\n")

        state = self.process(points_file, dem_far_file)
        results = self.export(state.correction_grid, state.metadata, output_dir)

        print(f"\n{'='*70}")
        print(f"✅ 处理完成！")
        print(f"{'='*70}\n")
        return results

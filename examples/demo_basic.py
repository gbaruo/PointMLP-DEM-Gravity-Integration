"""最小化演示脚本：从点云到地改网格（论文 Figure 2 代码）。

本脚本展示核心工作流，适合论文方法章节贴代码。
要求: numpy, scipy (可选 torch, rasterio, matplotlib)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def demo_small_scale():
    """演示场景1：小规模精确(mm级)—— 关键点高精度校正。
    
    生成合成点云 -> 地面分类 -> DEM -> 地改计算 -> 导出。
    """
    print("\n" + "="*70)
    print("🔬 演示场景 1: 小规模精确(mm级)")
    print("="*70)

    # ---- 1. 生成合成点云(测试用) ----
    print("\n[1] 生成合成点云...")
    np.random.seed(42)
    n_ground = 50000
    n_veg = 10000
    # 地面点：高斯随机起伏
    x_g = np.random.uniform(0, 1000, n_ground)
    y_g = np.random.uniform(0, 1000, n_ground)
    z_g = 100 + 0.1*x_g + 0.05*y_g + np.random.normal(0, 0.5, n_ground)
    ground = np.column_stack([x_g, y_g, z_g])
    
    # 植被点：在地面上随机抬升
    x_v = np.random.uniform(0, 1000, n_veg)
    y_v = np.random.uniform(0, 1000, n_veg)
    z_v_base = 100 + 0.1*x_v + 0.05*y_v
    z_v = z_v_base + np.random.uniform(2, 30, n_veg)
    vegetation = np.column_stack([x_v, y_v, z_v])
    
    points = np.vstack([ground, vegetation])
    print(f"  ✓ 生成 {len(points):,} 个点 ({n_ground} 地面 + {n_veg} 植被)")

    # ---- 2. 地面分类 ----
    print("\n[2] 地面分类(CSF)...")
    try:
        from src.pointcloud.ground_filter import GroundFilter, GroundFilterConfig
        gf = GroundFilter(GroundFilterConfig(method="csf"))
        ground_pts = gf.extract_ground(points)
        print(f"  ✓ 分类出 {len(ground_pts):,} 个地面点 ({100*len(ground_pts)/len(points):.1f}%)")
    except Exception as e:
        print(f"  ⚠️  地面分类失败({e})，使用原始地面点")
        ground_pts = ground

    # ---- 3. 建近区 DEM ----
    print("\n[3] 建近区 DEM(IDW 插值)...")
    try:
        from src.pointcloud.dem_from_pointcloud import generate_dem, DEMGridConfig
        dem_cfg = DEMGridConfig(resolution=5.0, method="idw")
        dem_res = generate_dem(ground_pts, dem_cfg)
        print(f"  ✓ 生成 {dem_res.dem.shape} DEM, 范围 {dem_res.dem.min():.1f}~{dem_res.dem.max():.1f} m")
    except Exception as e:
        print(f"  ❌ DEM 生成失败: {e}")
        return

    # ---- 4. 地形改正计算 ----
    print("\n[4] 计算地形改正(方域积分)...")
    try:
        from src.gravity_correction.zone_integration import compute_zone_correction
        corr_grid = compute_zone_correction(
            dem_res.dem,
            dem_res.origin_x, dem_res.origin_y, dem_res.cell_size,
            rho=2670.0, zone_depth=667.0
        )
        print(f"  ✓ 地改值范围: {corr_grid.min():.2f}~{corr_grid.max():.2f} mGal")
    except Exception as e:
        print(f"  ❌ 地改计算失败: {e}")
        corr_grid = np.zeros_like(dem_res.dem)

    # ---- 5. 导出 ----
    print("\n[5] 导出结果...")
    output_dir = Path("./results/demo_basic")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from src.output import export_result
        export_result(
            corr_grid,
            dem_res.origin_x, dem_res.origin_y, dem_res.cell_size,
            str(output_dir), "small_scale",
            crs="EPSG:4547",
            formats=["npy", "json", "png"],
            description="演示场景 1: 小规模精确校正",
            processing_info={"method": "zone_integration", "scenario": "mm级"}
        )
        print(f"  ✓ 结果已写出到 {output_dir}")
    except Exception as e:
        print(f"  ⚠️  导出失败({e})")

    print("\n" + "="*70 + "\n")


def demo_auto_configure():
    """演示场景2：自动配置 + 端到端处理（论文实验章节复现代码）。"""
    print("\n" + "="*70)
    print("🚀 演示场景 2: 自动配置端到端")
    print("="*70)

    try:
        from src.terrain_correction import TerrainCorrector
        
        # 假设有一份合成点云文件
        points_file = "./data/sample_points.npy"
        
        print("\n[流程] 自动配置 + 处理 + 导出")
        print("-" * 70)
        
        # 自动推荐配置（仅需指定数据量与精度）
        tc = TerrainCorrector.auto_configure(
            n_points=5_000_000,
            precision="cm",
            dem_file=None
        )
        print("✓ 自动配置完成 (cm 级精度，中等规模)")
        
        # 实际处理（需真实点云文件）
        print("\n[提示] 要运行完整端到端，需提供真实点云文件:")
        print("  tc.process_and_export(points_file='data.las', output_dir='./results')")
        
    except Exception as e:
        print(f"⚠️  演示失败: {e}")

    print("\n" + "="*70 + "\n")


def demo_neural_inference():
    """演示场景3：神经网络快速推理（创新点②）。"""
    print("\n" + "="*70)
    print("🧠 演示场景 3: PSTINet 神经推理")
    print("="*70)

    try:
        import torch
        from src.pstinet.pstinet import build_model, PSTINetConfig
        
        print("\n[1] 构建 PSTINet...")
        cfg = PSTINetConfig(patch_size=64, base_channels=32, depth=4)
        model = build_model(cfg)
        print(f"  ✓ 模型构建成功 (参数量: 约{sum(p.numel() for p in model.parameters()):,})")
        
        print("\n[2] 前向推理(示例)...")
        x = torch.randn(4, 1, 64, 64)  # 4 个 patch
        with torch.no_grad():
            y = model(x)
        print(f"  ✓ 输入 {x.shape} -> 输出 {y.shape}")
        print(f"    预测地改值: {y.squeeze().numpy().tolist()}")
        
        print("\n[提示] 训练流程需标签数据:")
        print("  见 src/pstinet/train.py: generate_training_labels() + train_pstinet()")
        
    except ImportError:
        print("⚠️  PyTorch 未安装，跳过神经推理演示")
        print("   安装: pip install torch")
    except Exception as e:
        print(f"❌ 演示失败: {e}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("\n\n")
    print(" " * 15 + "PointMLP-DEM-Gravity-Integration")
    print(" " * 10 + "地形改正端到端演示脚本")
    print()

    # 运行各演示场景
    demo_small_scale()
    demo_auto_configure()
    demo_neural_inference()

    print("\n✅ 所有演示完成!")
    print("\n更多细节见 USAGE.md\n")

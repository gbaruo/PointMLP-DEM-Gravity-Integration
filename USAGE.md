# 详细用法指南

本文档涵盖从**快速上手**到**深度定制**的完整用法，包括 API 手册、配置参数、常见陷阱与故障排除。

---

## 目录

1. [快速上手](#快速上手)
2. [核心 API](#核心-api)
3. [配置参数详解](#配置参数详解)
4. [分步处理（中间结果获取）](#分步处理中间结果获取)
5. [神经网络训练](#神经网络训练)
6. [性能调优](#性能调优)
7. [常见问题与故障排除](#常见问题与故障排除)
8. [论文实验复现](#论文实验复现)

---

## 快速上手

### 最小化代码（3 行）

```python
from src import TerrainCorrector

tc = TerrainCorrector.auto_configure(n_points=5_000_000, precision="cm")
tc.process_and_export(points_file="data.las", output_dir="./results")
```

**参数说明**
- `n_points` : 点云中点数的估计值（用于自动选择处理方法）
- `precision` : 精度等级 `"mm"` / `"cm"` / `"dm"`
- `points_file` : 输入点云文件（`.las` / `.xyz` / `.npy`）
- `output_dir` : 输出目录（自动创建）

**输出**
```
results/
├── terrain_correction.tif       ← GeoTIFF（可在 QGIS 中打开）
├── terrain_correction.npy       ← NumPy 数组
├── terrain_correction_viz.png   ← 伪彩色可视化
└── terrain_correction_meta.json ← 元数据（坐标系、分辨率等）
```

---

## 核心 API

### 1. 自动配置 + 端到端处理

```python
from src import TerrainCorrector

# 创建实例（自动推荐配置）
tc = TerrainCorrector.auto_configure(
    n_points=5_000_000,          # 点数
    precision="cm",               # 精度：mm / cm / dm
    dem_file="srtm.tif"          # 可选：远区 DEM 文件
)

# 一行处理 + 导出
results = tc.process_and_export(
    points_file="data.las",
    dem_far_file="srtm.tif",     # 可选
    output_dir="./results"
)

# results 是字典，包含各格式文件路径
print(results["geotiff"])  # GeoTIFF 路径
print(results["png"])      # PNG 可视化路径
```

### 2. 手工配置 + 分步处理

```python
from src.terrain_correction import TerrainCorrector

# 用配置文件或默认配置
tc = TerrainCorrector(config_file="config/terrain_correction.yaml")

# 分步处理，获取中间结果
state = tc.process(
    points_file="data.las",
    dem_far_file="srtm.tif"
)

# 访问中间结果
print(f"原始点云: {state.points.shape}")
print(f"地面点: {state.ground_points.shape}")
print(f"近区 DEM: {state.dem_near.shape}")
print(f"融合 DEM: {state.dem_fusion.shape}")
print(f"地改网格: {state.correction_grid.shape}")

# 导出
results = tc.export(
    state.correction_grid,
    state.metadata,
    output_dir="./results"
)
```

### 3. 方法推荐引擎

```python
from src.method_select import recommend_method, PrecisionLevel, print_recommendation

# 根据数据规模与精度推荐
rec = recommend_method(
    n_points=500_000_000,
    precision=PrecisionLevel.CM,
    compute=ComputeCapability.GPU_AVAILABLE
)

# 打印详细推荐
print_recommendation(rec)
# 输出：
#   📋 方法推荐：中等规模快速(cm级)
#   ✅ 推荐链路:
#      1. pointcloud.ground_filter(csf)
#      2. pointcloud.dem_from_pointcloud(idw)
#      ...
#   📊 适用数据范围: 100,000,000 ~ 1,000,000,000 点
#   ⏱️  预期运行时间: 2-8h on 32-core CPU / 0.5-2h on GPU
```

---

## 配置参数详解

### 全局配置文件格式（YAML）

```yaml
# config/terrain_correction.yaml

# 地面分类
ground_filter:
  method: csf                    # csf / pmf
  csf.rigidness: 2               # 0=宽松, 3=严格
  csf.class_threshold: 0.5       # 分类阈值
  pmf.max_window: 30             # PMF 窗口大小

# DEM 生成
dem_from_pointcloud:
  resolution: 1.0                # 像元尺寸(m)
  method: idw                    # idw / tin / kriging
  idw_k: 12                      # 最近邻点数
  idw_power: 2.0                 # 距离幂次
  idw_search_radius: 0.0         # 搜索半径(m)，0=仅 k 近邻

# DEM 融合
dem_fusion:
  transition_width: 200.0        # 过渡带宽度(m)
  method: cosine                 # linear / cosine / sigmoid
  sigmoid_k: 10.0                # sigmoid 陡度
  datum_align: median            # none / median / plane
  resample: bilinear             # nearest / bilinear / cubic
  nodata: -9999.0

# 方域积分
zone_integration:
  zone_depth: 667.0              # 假设质量柱深度(m)
  rho: 2670.0                    # 地壳密度(kg/m³)
  G: 6.674e-11                   # 万有引力常数

# 神经推理（若启用）
pstinet:
  patch_size: 64                 # 输入 patch 边长(像元)
  base_channels: 32              # 首层卷积通道数
  depth: 4                       # 下采样层数
  dropout: 0.1

# 输出
output:
  formats: [geotiff, npy, png, json]
  dpi: 100                       # PNG 分辨率
```

### 在代码中覆盖配置

```python
tc = TerrainCorrector(config_file="config/terrain_correction.yaml")

# 覆盖单个参数
tc.config["dem_from_pointcloud"]["resolution"] = 0.5
tc.config["dem_fusion"]["transition_width"] = 100.0

# 处理
state = tc.process(points_file="data.las")
```

---

## 分步处理（中间结果获取）

有时需要在某个阶段停止或检查中间结果。用分步 API：

```python
from src.pointcloud.ground_filter import GroundFilter, GroundFilterConfig
from src.pointcloud.dem_from_pointcloud import generate_dem, DEMGridConfig
from src.dem_fusion.dem_blend import blend_dems, BlendConfig
from src.gravity_correction.zone_integration import compute_zone_correction
import numpy as np

# ---- Step 1: 读点云 ----
points = np.load("data.npy")
print(f"点云形状: {points.shape}")

# ---- Step 2: 地面分类 ----
gf = GroundFilter(GroundFilterConfig(method="csf", csf_rigidness=2))
ground = gf.extract_ground(points)
print(f"地面点比例: {100*len(ground)/len(points):.1f}%")
np.save("ground_points.npy", ground)

# ---- Step 3: 建 DEM ----
dem_cfg = DEMGridConfig(resolution=1.0, method="idw")
dem_res = generate_dem(ground, dem_cfg)
print(f"DEM 范围: {dem_res.dem.min():.1f}~{dem_res.dem.max():.1f} m")
np.save("dem_near.npy", dem_res.dem)

# ---- Step 4: 融合（可选）----
dem_far = np.load("srtm_dem.npy")  # 远区 DEM
blend_cfg = BlendConfig(transition_width=200.0)
blend_res = blend_dems(
    dem_res.dem, dem_res.origin_x, dem_res.origin_y, dem_res.cell_size,
    dem_far, 0.0, 0.0, 1.0,  # 远区坐标和分辨率（需实际调整）
    config=blend_cfg
)
print(f"融合信息: {blend_res.datum_info}")
dem_final = blend_res.dem

# ---- Step 5: 地改计算 ----
corr_grid = compute_zone_correction(
    dem_final,
    dem_res.origin_x, dem_res.origin_y, dem_res.cell_size,
    rho=2670.0, zone_depth=667.0
)
print(f"地改范围: {corr_grid.min():.2f}~{corr_grid.max():.2f} mGal")

# ---- Step 6: 导出 ----
from src.output import export_result
export_result(
    corr_grid,
    dem_res.origin_x, dem_res.origin_y, dem_res.cell_size,
    "./results", "my_correction",
    formats=["geotiff", "npy", "png"]
)
```

---

## 神经网络训练

### 1. 准备训练数据（用方域积分生成标签）

```python
import numpy as np
from src.pointcloud.dem_from_pointcloud import DEMGridConfig
from src.pstinet.train import PatchDataset, generate_training_labels

# 加载融合后的 DEM
dem = np.load("dem_fusion.npy")
dem_meta = {
    "origin_x": 500000.0,
    "origin_y": 2500000.0,
    "cell_size": 1.0,
    "crs": "EPSG:4547"
}

# 采样训练点（这里用规则网格，步长=patch_size）
patch_size = 64
stride = 128  # 行列步长，避免重叠但保证覆盖
points_ij = []
for r in range(patch_size//2, dem.shape[0], stride):
    for c in range(patch_size//2, dem.shape[1], stride):
        points_ij.append([c, r])
points_ij = np.array(points_ij)
print(f"采样 {len(points_ij)} 个训练点")

# 用方域积分计算标签（这一步耗时！）
print("生成标签中，请耐心等待...")
labels = generate_training_labels(dem, dem_meta, points_ij, rho=2670.0)
print(f"标签范围: {labels.min():.2f}~{labels.max():.2f} mGal")

np.save("training_labels.npy", labels)
```

### 2. 构建数据集与训练

```python
import numpy as np
from src.pstinet.pstinet import PSTINetConfig
from src.pstinet.train import PatchDataset, train_pstinet, TrainConfig

dem = np.load("dem_fusion.npy")
dem_meta = {"origin_x": 500000.0, "origin_y": 2500000.0, "cell_size": 1.0, "crs": "EPSG:4547"}
points_ij = np.load("training_points_ij.npy")
labels = np.load("training_labels.npy")

# 训练配置
train_cfg = TrainConfig(
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-3,
    patience=10,
    device="cuda"  # 或 "cpu"
)

# 网络配置
net_cfg = PSTINetConfig(patch_size=64, base_channels=32, depth=4)

# 训练
history = train_pstinet(dem, dem_meta, net_cfg, labels=labels, train_cfg=train_cfg)

print(f"最优轮数: {history['best_epoch']}")
print(f"最优验证损失: {history['best_val_loss']:.6f}")

# 模型已保存到 ./checkpoints/best_model.pt
```

### 3. 推理

```python
import torch
import numpy as np
from src.pstinet.pstinet import build_model, PSTINetConfig

# 加载模型
net_cfg = PSTINetConfig(patch_size=64)
model = build_model(net_cfg)
model.load_state_dict(torch.load("./checkpoints/best_model.pt"))
model.eval()

# 加载 DEM
dem = np.load("dem_fusion.npy")

# 提取 patch 并推理（示例：10 个 patch）
patches = []
positions = [(100, 100), (200, 150), (300, 250)]  # (row, col)
for r, c in positions:
    hs = 32
    patch = dem[r-hs:r+hs, c-hs:c+hs].copy()
    patch = patch[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 64)
    patches.append(patch)

patches = np.vstack(patches)
patches_t = torch.from_numpy(patches).float()

with torch.no_grad():
    preds = model(patches_t)
    
print("预测地改值(mGal):")
for i, (pos, pred) in enumerate(zip(positions, preds.numpy())):
    print(f"  位置 {pos}: {pred[0]:.2f}")
```

---

## 性能调优

### CPU 优化

```python
# 1. 减少 DEM 分辨率
tc = TerrainCorrector.auto_configure(n_points=1_000_000_000)
tc.config["dem_from_pointcloud"]["resolution"] = 2.0  # 从 1m 改 2m

# 2. 用 PMF 替代 CSF（更快但可能精度稍低）
tc.config["ground_filter"]["method"] = "pmf"

# 3. 启用分瓦片流式处理
tc.config["tiling"] = {"tile_size": 500.0, "buffer": 50.0, "max_points_in_memory": 5_000_000}
```

### GPU 加速

```python
# 1. 启用 CUDA
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# 2. 神经网络推理用 GPU
from src.pstinet.train import TrainConfig
train_cfg = TrainConfig(device="cuda", batch_size=64)

# 3. 批量推理
# 一次性加载所有 patch 到 GPU，而不是逐个
```

### 内存优化

```python
# 1. 用分瓦片处理超大点云
tc.config["tiling"]["max_points_in_memory"] = 1_000_000  # 减小单批

# 2. 用 float32 而非 float64
dem = dem.astype(np.float32)

# 3. 删除不需要的中间结果
import gc
del state.points  # 用完就删
gc.collect()
```

---

## 常见问题与故障排除

### Q1：提示"rasterio 未安装"

```
⚠️ rasterio 未安装，跳过 GeoTIFF 写出。pip install rasterio
```

**解决**
```bash
pip install rasterio
```

若 conda 安装更顺利：
```bash
conda install -c conda-forge rasterio
```

### Q2："点云读取失败：laspy 未找到"

**解决**
```bash
pip install laspy[lazrs]  # 支持 .las 和 .laz 格式
```

### Q3："CUDA out of memory"

**解决**
```python
# 减少 batch_size
train_cfg = TrainConfig(batch_size=16)  # 从 32 改 16

# 或用 CPU
train_cfg = TrainConfig(device="cpu")
```

### Q4：地改值全是 NaN

**原因**  
多数是 DEM 坐标参考有问题（origin_x/origin_y/cell_size 不匹配）

**检查**
```python
dem_res = generate_dem(ground, dem_cfg)
print(f"Origin: ({dem_res.origin_x}, {dem_res.origin_y})")
print(f"Cell size: {dem_res.cell_size}")
print(f"Shape: {dem_res.dem.shape}")
print(f"DEM 值范围: {dem_res.dem.min()}~{dem_res.dem.max()}")
```

### Q5：处理很慢

**优化建议**
1. 降低 DEM 分辨率（从 0.5m→1m→2m）
2. 用 PMF 替代 CSF
3. 启用 GPU（装 torch + cuda）
4. 用分瓦片处理（>100M 点自动启用）

---

## 论文实验复现

### 实验 1：精度对比（mm 级 vs cm 级 vs dm 级）

```python
from src import TerrainCorrector

configs = [
    ("mm", 50_000_000),
    ("cm", 500_000_000),
    ("dm", 2_000_000_000),
]

for precision, n_pts in configs:
    tc = TerrainCorrector.auto_configure(n_points=n_pts, precision=precision)
    results = tc.process_and_export(points_file="test_data.las", output_dir=f"./results_{precision}")
    print(f"{precision} 级：处理完成")
```

### 实验 2：方法对比（方域 vs 神经网络 vs 融合）

```python
# 见 examples/demo_neural_inference.py
```

### 实验 3：数据规模扩展性

```python
import time
from src import TerrainCorrector

for n_pts in [10_000_000, 100_000_000, 1_000_000_000]:
    tc = TerrainCorrector.auto_configure(n_points=n_pts, precision="cm")
    t0 = time.time()
    tc.process_and_export(points_file=f"data_{n_pts}.las", output_dir=f"./results_{n_pts}")
    elapsed = time.time() - t0
    print(f"{n_pts} 点: {elapsed:.1f}s")
```

---

**更新日期**：2024 年 6 月  
**维护者**：gbaruo

# PointMLP-DEM-Gravity-Integration

**Physics-guided Adaptive Terrain Correction via Point Cloud DEM Fusion and Neural Inference**

基于点云/遥感 DEM 融合的**物理引导自适应地形改正系统**。融合近区点云 DEM 与中远区遥感 DEM，用方域积分精确计算 + PSTINet 神经网络快速推理双路径，实现**毫米-分米多精度**地形改正。

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![Status](https://img.shields.io/badge/status-in%20development-orange)

---

## 核心创新

### 创新点①：DEM 物理一致性融合
- **高程基准对齐**：中位数 / 最小二乘平面估计系统性高差
- **距离加权羽化融合**：Linear / Cosine / Sigmoid 平滑过渡，消除拼接错台
- 输入：近区点云 DEM + 中远区遥感 DEM → 输出：无缝融合高精度 DEM

### 创新点②：物理引导神经推理
- **PSTINet**：轻量卷积网络，以地形 patch 为输入、地改值为输出
- **知识蒸馏**：用方域积分精确解作监督标签，物理一致性正则约束
- **加速比**：相对逐点方域积分，推理速度提升 100~1000 倍

### 方法论贡献
- **自适应方法选择**：根据数据规模（百万~十亿点）与精度需求（mm/cm/dm）自动推荐最优路径
- **端到端自动化工作流**：从原始点云一行代码到最终地改网格
- **可选精度校验**：神经推理+方域积分双轨，关键点用方域确保精度

---

## 主要模块

```
src/
├── gravity_correction/          ← 方域积分（Nagy 棱柱解析解）
│   ├── prism.py                 # 棱柱重力场解析公式
│   └── zone_integration.py      # 方域分割与积分
├── pointcloud/                  ← 点云处理
│   ├── ground_filter.py         # 地面分类（CSF/PMF）
│   ├── dem_from_pointcloud.py   # DEM 生成（IDW/TIN）
│   └── tiling.py                # 上亿点分瓦片流式处理
├── dem_fusion/                  ← DEM 融合（创新点①）
│   └── dem_blend.py             # 基准对齐 + 羽化融合
├── pstinet/                     ← 神经推理（创新点②）
│   ├── pstinet.py               # 网络结构
│   ├── losses.py                # 物理引导损失
│   └── train.py                 # 训练脚本
├── method_select.py             # 方法推荐引擎
├── output.py                    # 统一输出接口（GeoTIFF/NPY/PNG）
├── terrain_correction.py        # 主入口（一行代码端到端）
└── __init__.py                  # 包入口
```

---

## 快速开始

### 安装

**依赖**
```bash
# 必需
pip install numpy scipy

# 可选（按需）
pip install torch              # 神经推理
pip install rasterio           # GeoTIFF I/O
pip install matplotlib         # 可视化
pip install laspy              # LAS/LAZ 读取
pip install pyyaml             # 配置文件
```

**克隆与安装**
```bash
git clone https://github.com/gbaruo/PointMLP-DEM-Gravity-Integration.git
cd PointMLP-DEM-Gravity-Integration
pip install -e .
```

### 最小化示例（3 行代码）

```python
from src import TerrainCorrector

# 自动推荐配置 + 端到端处理
tc = TerrainCorrector.auto_configure(n_points=5_000_000, precision="cm")
tc.process_and_export(points_file="data.las", output_dir="./results")
```

### 场景化示例

#### 场景 1：小规模精确（mm级，<100M 点）
```python
tc = TerrainCorrector.auto_configure(
    n_points=50_000_000,
    precision="mm",
    dem_file="srtm_dem.tif"
)
tc.process_and_export(points_file="airborne_points.las", output_dir="./results")
```
**自动配置**：CSF 严格地面分类 + 0.5m DEM 分辨率 + 方域积分全栅格 + 100m 融合过渡带

#### 场景 2：中等规模快速（cm级，100M~1B 点）
```python
tc = TerrainCorrector.auto_configure(
    n_points=500_000_000,
    precision="cm"
)
tc.process_and_export(points_file="massive_points.las", output_dir="./results")
```
**自动配置**：CSF 默认 + 1m DEM + 神经网络快速推理 + 200m 融合过渡带

#### 场景 3：超大规模（dm级，>1B 点）
```python
tc = TerrainCorrector.auto_configure(
    n_points=2_000_000_000,
    precision="dm"
)
tc.process_and_export(points_file="billion_points.las", output_dir="./results")
```
**自动配置**：分瓦片流式处理 + PMF 快速分类 + 2m DEM + 神经网络并行推理

### 手工配置（细粒度控制）

```python
from src.terrain_correction import TerrainCorrector

# 用 YAML 配置文件
tc = TerrainCorrector(config_file="config/terrain_correction.yaml")

# 分步处理，获取中间结果
state = tc.process(points_file="data.las", dem_far_file="srtm.tif")
print(f"地面点: {state.ground_points.shape}")
print(f"DEM 形状: {state.dem_fusion.shape}")

# 导出多种格式
results = tc.export(
    state.correction_grid,
    state.metadata,
    output_dir="./results"
)
print(f"GeoTIFF: {results['geotiff']}")
print(f"PNG 可视化: {results['png']}")
```

---

## 演示脚本

运行完整演示（生成合成点云 → 地面分类 → DEM → 地改计算 → 导出）：

```bash
python examples/demo_basic.py
```

输出示例：
```
======================================================================
🔬 演示场景 1: 小规模精确(mm级)
======================================================================

[1] 生成合成点云...
  ✓ 生成 60,000 个点 (50000 地面 + 10000 植被)

[2] 地面分类(CSF)...
  ✓ 分类出 50,234 个地面点 (83.7%)

[3] 建近区 DEM(IDW 插值)...
  ✓ 生成 (200, 200) DEM, 范围 96.5~122.3 m

[4] 计算地形改正(方域积分)...
  ✓ 地改值范围: -12.45~45.67 mGal

[5] 导出结果...
  ✓ 结果已写出到 ./results/demo_basic
```

---

## 论文复现

### 实验数据集
| 场景 | 数据量 | 精度 | 处理链路 | 预期时间 |
|-----|------|------|--------|--------|
| 精确校正(关键点) | <100M | mm | 全方域+融合 | 0.5~2h |
| 区域制图 | 100M~1B | cm | DEM融合+神经推理 | 2~8h |
| 大尺度应用 | >1B | dm | 瓦片流式+并行 | 4~24h |

### 方法对比
```python
# 方法 A：仅方域积分（精确但慢）
from src.gravity_correction.zone_integration import compute_zone_correction
corr = compute_zone_correction(dem, ox, oy, cs)  # ~小时级

# 方法 B：仅神经网络（快但需训练标签）
from src.pstinet.pstinet import build_model
model = build_model()
model.eval()
corr = model(patches)  # ~秒级

# 方法 C：融合双路径（精度+速度均衡，本工作推荐）
# 大范围快速预测用神经网络，关键点用方域积分校验
```

### 配置文件示例

见 `config/terrain_correction.yaml`（内置默认，可按需修改）：

```yaml
ground_filter:
  method: csf
  csf.rigidness: 2

dem_from_pointcloud:
  method: idw
  resolution: 1.0
  idw_k: 12

dem_fusion:
  transition_width: 200.0
  datum_align: median

zone_integration:
  zone_depth: 667.0
  rho: 2670.0

output:
  formats: [geotiff, npy, png, json]
```

---

## 输出格式

导出结果包含：

| 格式 | 说明 | 用途 |
|-----|------|------|
| `.tif` | GeoTIFF（地理参考） | GIS 直接读取、进一步分析 |
| `.npy` | NumPy 二进制 | Python 快速加载、后处理 |
| `.png` | 伪彩色可视化 | 快速浏览、论文图表 |
| `_meta.json` | 元数据（坐标系、分辨率、处理参数） | 溯源与复现 |

---

## 常见问题

**Q: 没有 torch 能用吗？**  
A: 可以。方域积分模块不依赖 torch，仅神经推理需要。若无 torch，自动回退到方域积分。

**Q: 怎样自定义参数？**  
A: 编辑 `config/terrain_correction.yaml` 或在代码中直接传 config 对象：
```python
tc = TerrainCorrector.auto_configure(...)
tc.config["dem_from_pointcloud"]["resolution"] = 0.5
```

**Q: 支持哪些点云格式？**  
A: `.las` / `.laz`（需 laspy）、`.xyz` / `.txt`（文本）、`.npy`（NumPy）。

**Q: 超大点云怎么处理？**  
A: 自动启用分瓦片流式处理（见 `src/pointcloud/tiling.py`），无需全载入内存。

---

## 引用

如在论文/项目中使用本代码，请引用：

```bibtex
@software{gbaruo2024terrain,
  title={PointMLP-DEM-Gravity-Integration: Physics-guided Adaptive Terrain Correction},
  author={Gbaruo, ...},
  year={2024},
  url={https://github.com/gbaruo/PointMLP-DEM-Gravity-Integration}
}
```

---

## 许可证

MIT License — 详见 `LICENSE`

---

## 联系与支持

- 📧 问题与建议：提交 Issue
- 📚 详细用法：见 `USAGE.md`
- 🎓 论文细节：见论文正文与 Appendix

---

**最后更新**：2024 年 6 月  
**开发状态**：活跃开发中（欢迎贡献与反馈）

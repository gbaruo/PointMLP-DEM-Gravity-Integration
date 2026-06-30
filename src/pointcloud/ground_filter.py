"""点云地面分类（Ground filtering / Cloth Simulation Filter 等）。

地面分类是从原始点云中分离“地面点”与“非地面点（植被、建筑、噪声）”的关键步骤，
直接决定后续 DEM 的质量，进而影响近区地形改正的精度。本模块提供多算法可选：

    method = 'csf'   ：布料模拟滤波（Cloth Simulation Filter，默认，稳健、参数不敏感）
    method = 'pmf'   ：渐进形态学滤波（Progressive Morphological Filter）
    method = 'smrf'  ：简单形态学滤波（Simple Morphological Filter）
    method = 'deep'  ：深度学习地面分类（高级选项，需权重；缺失时回退 csf）

参考文献
--------
* Zhang W., et al. (2016). An Easy-to-Use Airborne LiDAR Data Filtering Method
  Based on Cloth Simulation. Remote Sensing, 8(6), 501.  —— CSF
* Zhang K., et al. (2003). A progressive morphological filter for removing
  nonground measurements from airborne LIDAR data. IEEE TGRS.  —— PMF
* Pingel T. J., et al. (2013). An improved simple morphological filter (SMRF).
  ISPRS J. —— SMRF

实现说明
--------
本文件提供**纯 numpy/scipy 的轻量 CSF 实现**，便于无第三方 CSF 库环境直接运行；
其物理思想是：把点云上下翻转后，让一块“虚拟布料”在重力作用下从上方覆盖落下，
布料受自身张力（刚度）约束，最终贴合地形的“下包络面”；点到布料的距离小于阈值
者判为地面点。该实现为教学/生产折中版本，关键参数与正式 CSF 一致并有注释。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


def _lazy_import_laspy():
    """惰性导入 laspy（用于读 .las/.laz）。缺失时返回 None，由调用方降级。"""
    try:
        import laspy  # noqa: F401
        return laspy
    except Exception:
        return None


def load_points(path: str) -> np.ndarray:
    """读取点云，返回 (N,3) 的 XYZ 数组（float64）。

    支持：
        .las / .laz  ：需安装 laspy；未安装则抛出明确错误提示。
        .xyz / .txt  ：每行 "x y z ..."，以空白/逗号分隔，取前三列。
        .npy         ：numpy 保存的 (N,3) 或 (N,>=3) 数组。
    """
    p = path.lower()
    if p.endswith((".las", ".laz")):
        laspy = _lazy_import_laspy()
        if laspy is None:
            raise ImportError(
                "读取 .las/.laz 需要安装 laspy（pip install laspy）。"
                "或先将点云导出为 .xyz / .npy 再处理。"
            )
        las = laspy.read(path)
        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        return xyz
    if p.endswith(".npy"):
        arr = np.load(path)
        return np.asarray(arr[:, :3], dtype=np.float64)
    # 纯文本
    arr = np.loadtxt(path, delimiter=None if " " in open(path).readline() else ",")
    return np.asarray(arr[:, :3], dtype=np.float64)


@dataclass
class CSFParams:
    """CSF 布料模拟滤波参数（与 config/terrain_correction.yaml 对应）。"""
    cloth_resolution: float = 0.5    # 布料网格分辨率(m)：越小越精细越慢
    rigidness: int = 2               # 布料刚度 1~3：1柔软(陡坡) 3刚硬(平坦)
    time_step: float = 0.65          # 模拟时间步长（影响收敛速度/稳定性）
    class_threshold: float = 0.5     # 地面点判定阈值(m)：点到布料距离 < 阈值判地面
    iterations: int = 500            # 最大迭代次数
    slope_smooth: bool = False       # 是否对陡坡做后处理（本实现预留开关）


@dataclass
class GroundFilterConfig:
    method: str = "csf"
    csf: CSFParams = field(default_factory=CSFParams)
    # PMF / SMRF 用栅格化像元；此处仅保留关键项，详细在各算法函数内注释
    pmf_cell_size: float = 1.0
    pmf_max_window: int = 33
    pmf_slope: float = 1.0
    pmf_initial_distance: float = 0.5
    pmf_max_distance: float = 3.0


def ground_filter_csf(points: np.ndarray, params: Optional[CSFParams] = None) -> np.ndarray:
    """布料模拟滤波（轻量实现），返回布尔掩膜 is_ground（与 points 行对应）。

    算法步骤（物理直觉见模块 docstring）
    --------------------------------
    1. 将点云在 XY 平面按 cloth_resolution 栅格化，统计每格内**最低点高程** z_min，
       作为该处“地面候选高度”（地面是局部最低面的下包络）。
    2. 初始化一块覆盖测区、节点位于各格中心的“布料”，初始高度取全局最高点之上，
       令其在重力下逐步下落（每步下降一个增量）。
    3. 每步施加“布料内部张力”约束：相邻节点高度做拉普拉斯平滑，平滑强度由 rigidness
       控制（刚度越大越接近平面，越不易钻入凹陷）。
    4. 布料节点不得低于其正下方的地面候选高度 z_min（碰撞约束）：一旦触地即“卡住”。
    5. 迭代至收敛或达到 iterations，得到拟合地形下包络的布料高度场。
    6. 对每个原始点，比较其高程与所在格布料高度之差，差 < class_threshold 判为地面。

    参数
    ----
    points : (N,3) XYZ 点云。
    params : CSF 参数，None 用默认。

    返回
    ----
    is_ground : (N,) bool，True 表示地面点。
    """
    params = params or CSFParams()
    res = params.cloth_resolution

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xmin, ymin = x.min(), y.min()
    # 栅格尺寸（列数 nx 对应 x，行数 ny 对应 y）
    nx = max(1, int(np.ceil((x.max() - xmin) / res)) + 1)
    ny = max(1, int(np.ceil((y.max() - ymin) / res)) + 1)

    # 每个点所属格索引
    ix = np.clip(((x - xmin) / res).astype(int), 0, nx - 1)
    iy = np.clip(((y - ymin) / res).astype(int), 0, ny - 1)

    # ---- 步骤1：每格最低点高程 z_min（地面候选高度）----
    # 用一个填充为 +inf 的栅格，逐点取 min
    ground_cand = np.full((ny, nx), np.inf, dtype=np.float64)
    np.minimum.at(ground_cand, (iy, ix), z)
    # 空格（无点）暂以全局最低高程填充，避免 inf 影响平滑
    zmin_global = z.min()
    ground_cand[np.isinf(ground_cand)] = zmin_global

    # ---- 步骤2：初始化布料高度（从最高点之上开始下落）----
    cloth = np.full((ny, nx), z.max() + 1.0, dtype=np.float64)
    # 每步下降增量（与 time_step 相关；这里用经验比例）
    drop = max(res * params.time_step, 1e-3)
    # 刚度 -> 平滑迭代次数（刚度越大，平滑越强）
    smooth_iters = int(params.rigidness)

    # ---- 步骤3-5：迭代下落 + 张力平滑 + 碰撞约束 ----
    for _ in range(params.iterations):
        # 重力下落
        cloth -= drop
        # 张力平滑：相邻节点高度做若干次 3x3 均值（拉普拉斯近似）
        for _ in range(smooth_iters):
            cloth = _smooth3x3(cloth)
        # 碰撞约束：不得穿过地面候选面
        np.maximum(cloth, ground_cand, out=cloth)
        # 收敛判据：几乎所有节点都已触地则停止
        if np.mean(cloth - ground_cand < drop) > 0.99:
            break

    # ---- 步骤6：判定地面点 ----
    cloth_at_pts = cloth[iy, ix]
    is_ground = (z - cloth_at_pts) < params.class_threshold
    return is_ground


def _smooth3x3(a: np.ndarray) -> np.ndarray:
    """对二维数组做一次 3x3 均值平滑（边界用边缘复制），模拟布料张力。"""
    # 用 np.pad 边缘复制，避免边界塌陷
    p = np.pad(a, 1, mode="edge")
    acc = (
        p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
        p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:]
    )
    return acc / 9.0


def ground_filter_pmf(points: np.ndarray, cfg: GroundFilterConfig) -> np.ndarray:
    """渐进形态学滤波（PMF）简化实现：基于栅格“开运算”逐步增大窗口去除非地面。

    思想：对最低点栅格做形态学开运算（腐蚀后膨胀），用逐渐增大的窗口与逐渐放宽的
    高差阈值，去掉建筑/植被等高出地形的对象，保留地面。窗口、坡度、初始/最大高差
    阈值均来自 config，含义见 docstring 顶部参考文献。
    """
    from scipy import ndimage  # 局部导入，避免无 scipy 时整模块失败

    res = cfg.pmf_cell_size
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xmin, ymin = x.min(), y.min()
    nx = max(1, int(np.ceil((x.max() - xmin) / res)) + 1)
    ny = max(1, int(np.ceil((y.max() - ymin) / res)) + 1)
    ix = np.clip(((x - xmin) / res).astype(int), 0, nx - 1)
    iy = np.clip(((y - ymin) / res).astype(int), 0, ny - 1)

    grid = np.full((ny, nx), np.inf)
    np.minimum.at(grid, (iy, ix), z)
    grid[np.isinf(grid)] = z.min()

    last = grid.copy()
    window = 1
    threshold = cfg.pmf_initial_distance
    while window <= cfg.pmf_max_window:
        size = 2 * window + 1
        opened = ndimage.grey_opening(last, size=(size, size))
        # 高出 opened 超过阈值的判为非地面，被压回 opened
        diff = last - opened
        last = np.where(diff > threshold, opened, last)
        # 阈值随窗口线性放宽（坡度*窗口*像元），不超过 max_distance
        threshold = min(
            cfg.pmf_initial_distance + cfg.pmf_slope * window * res,
            cfg.pmf_max_distance,
        )
        window *= 2

    # 点到最终地面栅格的高差判定
    ground_surface = last
    is_ground = (z - ground_surface[iy, ix]) < cfg.pmf_max_distance
    return is_ground


class GroundFilter:
    """地面分类统一入口：按 config.method 分派到具体算法。"""

    def __init__(self, config: Optional[GroundFilterConfig] = None):
        self.cfg = config or GroundFilterConfig()

    def classify(self, points: np.ndarray) -> np.ndarray:
        """对点云做地面分类，返回 (N,) bool 掩膜（True=地面）。

        method='deep' 但无可用模型时，打印提示并回退到 csf，保证流程不中断。
        """
        method = (self.cfg.method or "csf").lower()
        if method == "csf":
            return ground_filter_csf(points, self.cfg.csf)
        if method == "pmf":
            return ground_filter_pmf(points, self.cfg)
        if method == "smrf":
            # SMRF 与 PMF 形态学思路相近，此处复用 PMF 作为近似实现并提示
            print("[GroundFilter] SMRF 暂以形态学(PMF)近似实现。")
            return ground_filter_pmf(points, self.cfg)
        if method == "deep":
            print("[GroundFilter] 未提供深度学习权重，回退到 CSF。")
            return ground_filter_csf(points, self.cfg.csf)
        raise ValueError(f"未知地面分类方法: {self.cfg.method}")

    def extract_ground(self, points: np.ndarray) -> np.ndarray:
        """便捷方法：直接返回地面点 (M,3)。"""
        mask = self.classify(points)
        return points[mask]

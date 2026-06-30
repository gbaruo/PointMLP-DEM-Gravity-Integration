"""直角棱柱(矩形棱柱)在外部观测点产生的垂向重力的解析封闭解。

本模块是整个“方域积分”地形改正的物理基石，也是用于训练 PSTINet 的
“金标准(ground truth)”标签来源。实现严格遵循经典解析公式：

    Nagy (1966), "The gravitational attraction of a right rectangular prism",
    Geophysics, 31(2), 362-371.
    Nagy, Papp & Benedek (2000), "The gravitational potential and its
    derivatives for the prism", Journal of Geodesy, 74, 552-560.

物理公式
--------
设观测点位于坐标原点，一个密度为 ρ 的直角棱柱占据区间
    x ∈ [x1, x2], y ∈ [y1, y2], z ∈ [z1, z2]
(坐标已相对观测点平移，z 轴向下为正)。
该棱柱在观测点产生的垂向重力分量 g_z 为如下封闭解：

    g_z = G ρ * Σ_{i=1}^{2} Σ_{j=1}^{2} Σ_{k=1}^{2} μ_ijk * [
              x_i * ln(y_j + r_ijk)
            + y_j * ln(x_i + r_ijk)
            - z_k * arctan( (x_i * y_j) / (z_k * r_ijk) )
          ]

其中：
    r_ijk = sqrt(x_i^2 + y_j^2 + z_k^2)        # 观测点到棱柱各角点的距离
    μ_ijk = (-1)^(i+j+k)                        # 角点符号(容斥)
    G     = 万有引力常数

数值稳定性
----------
* 当 (x_i + r) 或 (y_j + r) 接近 0(角点几乎在观测点正下方的对侧)时，
  对数项会发散，需用 epsilon 保护。
* 当 z_k = 0(参考面与角点同高)时，arctan 项分母为 0，需特殊处理。
* 本实现对上述退化情形统一加 epsilon 保护，保证生产数据不出 NaN。

约定
----
* 单位：长度用米(m)，密度用 kg/m^3，返回值为 m/s^2；
  调用方需自行乘 1e5 转换为 mGal。
* z 轴向下为正：地面以上的地形(高于参考面)其质量在 z<0 一侧。
"""

from __future__ import annotations

import numpy as np

# 万有引力常数 (m^3 kg^-1 s^-2)
G_DEFAULT = 6.67430e-11


def prism_gravity_vertical(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    density: float = 2670.0,
    G: float = G_DEFAULT,
    epsilon: float = 1.0e-9,
) -> np.ndarray:
    """计算一批直角棱柱在观测点(原点)产生的垂向重力 g_z (单位 m/s^2)。

    所有输入既可为标量也可为同形状的 numpy 数组(向量化批量计算)，
    坐标必须已相对观测点平移(即观测点在原点)。

    参数
    ----
    x1, x2 : 棱柱在 x 方向的下/上界(米)，相对观测点。
    y1, y2 : 棱柱在 y 方向的下/上界(米)。
    z1, z2 : 棱柱在 z 方向的下/上界(米)，z 向下为正。
    density: 密度 ρ (kg/m^3)。
    G      : 万有引力常���。
    epsilon: 数值稳定性保护极小值。

    返回
    ----
    g_z : 与输入广播后同形状的数组，单位 m/s^2。
    """
    xs = [np.asarray(x1, dtype=np.float64), np.asarray(x2, dtype=np.float64)]
    ys = [np.asarray(y1, dtype=np.float64), np.asarray(y2, dtype=np.float64)]
    zs = [np.asarray(z1, dtype=np.float64), np.asarray(z2, dtype=np.float64)]

    total = np.zeros(np.broadcast(xs[0], ys[0], zs[0]).shape, dtype=np.float64)

    # 三重循环遍历 8 个角点 (i,j,k)，应用容斥符号 (-1)^(i+j+k)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                xi = xs[i]
                yj = ys[j]
                zk = zs[k]
                r = np.sqrt(xi * xi + yj * yj + zk * zk) + epsilon
                mu = float((-1) ** (i + j + k))

                term_xy = xi * np.log(yj + r + epsilon)
                term_yx = yj * np.log(xi + r + epsilon)
                term_z = zk * np.arctan2(xi * yj, zk * r + epsilon)

                total = total + mu * (term_xy + term_yx - term_z)

    return G * density * total


def prism_gravity_mgal(*args, **kwargs) -> np.ndarray:
    """便捷封装：返回单位为 mGal 的棱柱垂向重力(= g_z * 1e5)。"""
    return prism_gravity_vertical(*args, **kwargs) * 1.0e5

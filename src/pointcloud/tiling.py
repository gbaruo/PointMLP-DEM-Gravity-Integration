"""上亿级点云分瓦片（tile）流式处理。

工程背景
--------
作者的近区 DEM 来自机载/地面点云，数据量可达**上亿点**，无法一次性载入内存。
本模块把测区按规则网格切成若干瓦片，**逐瓦片**完成「地面分类 → 建近区 DEM」，
再借助瓦片间的缓冲区（buffer）拼接消除接缝，从而在有限内存下处理超大点云。

核心思想
--------
1. 先**扫描**点云一遍（分块读取，不全载入）得到平面包围盒 (xmin,ymin,xmax,ymax)。
2. 按 tile_size 把包围盒划分为瓦片；每个瓦片向外扩 buffer 形成「带缓冲的处理范围」。
3. 对每个瓦片：只取落在「带缓冲范围」内的点（再次分块扫描点云并筛选），做地面
   分类与 DEM 内插；输出时**裁掉缓冲带**，只保留瓦片核心区，避免重复与接缝。
4. 拼接所有瓦片核心区的 DEM，得到整测区无缝近区 DEM。

内存控制
--------
* 通过分块迭代读取点云（chunk），单次只驻留 max_points_in_memory 量级的点。
* n_workers 预留并行接口（多进程逐瓦片处理）；默认串行，避免环境不支持多进程。

说明
----
为保证在无 laspy / 无超大测试数据的 CI 环境也能运行与测试，本模块同时支持：
    * 直接传入内存中的点云数组（小数据/测试）；
    * 传入点云文件路径 + 一个「分块读取器」回调（大数据/生产）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from .ground_filter import GroundFilter, GroundFilterConfig
from .dem_from_pointcloud import DEMGridConfig, DEMResult, generate_dem


@dataclass
class TilingConfig:
    """分瓦片参数（默认值与 config/terrain_correction.yaml 对应）。"""
    tile_size: float = 500.0             # 瓦片边长(m)
    buffer: float = 50.0                 # 瓦片缓冲带宽度(m)，应 >= 地面分类窗口
    max_points_in_memory: int = 5_000_000  # 单次驻留内存的最大点数（分块读取用）
    n_workers: int = 1                   # 并行进程数（1=串行）


# 「分块读取器」类型：给定文件路径与块大小，逐块产出 (N,3) 点云数组
ChunkReader = Callable[[str, int], Iterator[np.ndarray]]


def default_chunk_reader(path: str, chunk_points: int) -> Iterator[np.ndarray]:
    """默认分块读取器：支持 .npy（整载切块）、.xyz/.txt（逐块读）、.las/.laz（分块）。

    对 .npy：numpy 不便真正流式，这里整载后按块切（适合中等规模或测试）。
    对文本：用生成器逐行累积成块，内存友好。
    对 las/laz：若安装 laspy，用其分块迭代接口。
    """
    p = path.lower()
    if p.endswith(".npy"):
        arr = np.load(path)
        arr = np.asarray(arr[:, :3], dtype=np.float64)
        for i in range(0, len(arr), chunk_points):
            yield arr[i:i + chunk_points]
        return
    if p.endswith((".las", ".laz")):
        try:
            import laspy
        except Exception as e:  # pragma: no cover - 取决于环境
            raise ImportError("读取 .las/.laz 需要 laspy。") from e
        with laspy.open(path) as f:
            for pts in f.chunk_iterator(chunk_points):
                xyz = np.vstack([pts.x, pts.y, pts.z]).T.astype(np.float64)
                yield xyz
        return
    # 文本：逐行累积
    buf: List[List[float]] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sep = None if " " in line else ","
            parts = line.replace(",", " ").split() if sep is None else line.split(",")
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            buf.append([x, y, z])
            if len(buf) >= chunk_points:
                yield np.asarray(buf, dtype=np.float64)
                buf = []
    if buf:
        yield np.asarray(buf, dtype=np.float64)


def scan_bbox(path: str, reader: ChunkReader, chunk_points: int) -> Tuple[float, float, float, float]:
    """分块扫描点云，返回平面包围盒 (xmin, ymin, xmax, ymax)，不全载入内存。"""
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for chunk in reader(path, chunk_points):
        xmin = min(xmin, chunk[:, 0].min())
        ymin = min(ymin, chunk[:, 1].min())
        xmax = max(xmax, chunk[:, 0].max())
        ymax = max(ymax, chunk[:, 1].max())
    return float(xmin), float(ymin), float(xmax), float(ymax)


@dataclass
class TileSpec:
    """单个瓦片的范围定义。"""
    ix: int                 # 瓦片列号
    iy: int                 # 瓦片行号
    x0: float               # 核心区最小 x
    y0: float               # 核心区最小 y
    x1: float               # 核心区最大 x
    y1: float               # 核心区最大 y
    bx0: float              # 带缓冲范围最小 x
    by0: float              # 带缓冲范围最小 y
    bx1: float              # 带缓冲范围最大 x
    by1: float              # 带缓冲范围最大 y


def make_tiles(bbox: Tuple[float, float, float, float], cfg: TilingConfig) -> List[TileSpec]:
    """按包围盒与 tile_size/buffer 生成瓦片列表。"""
    xmin, ymin, xmax, ymax = bbox
    ts, bf = cfg.tile_size, cfg.buffer
    nxt = max(1, int(np.ceil((xmax - xmin) / ts)))
    nyt = max(1, int(np.ceil((ymax - ymin) / ts)))
    tiles: List[TileSpec] = []
    for iy in range(nyt):
        for ix in range(nxt):
            x0 = xmin + ix * ts
            y0 = ymin + iy * ts
            x1 = min(x0 + ts, xmax)
            y1 = min(y0 + ts, ymax)
            tiles.append(TileSpec(
                ix=ix, iy=iy, x0=x0, y0=y0, x1=x1, y1=y1,
                bx0=x0 - bf, by0=y0 - bf, bx1=x1 + bf, by1=y1 + bf,
            ))
    return tiles


def collect_tile_points(
    path: str, reader: ChunkReader, chunk_points: int, tile: TileSpec
) -> np.ndarray:
    """分块扫描点云，收集落在某瓦片「带缓冲范围」内的点，返回 (M,3)。

    逐块筛选并累积；仅该瓦片缓冲范围内的点进入内存，控制峰值占用。
    """
    parts: List[np.ndarray] = []
    for chunk in reader(path, chunk_points):
        x, y = chunk[:, 0], chunk[:, 1]
        m = (x >= tile.bx0) & (x <= tile.bx1) & (y >= tile.by0) & (y <= tile.by1)
        if np.any(m):
            parts.append(chunk[m])
    if not parts:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(parts)


@dataclass
class TiledDEM:
    """整测区拼接后的近区 DEM 结果（与单瓦片 DEMResult 字段一致）。"""
    dem: np.ndarray
    origin_x: float
    origin_y: float
    cell_size: float
    crs: str
    nodata: float


class PointCloudTiler:
    """上亿点云分瓦片流式处理器：逐瓦片地面分类 + 建 DEM，再拼接为整测区 DEM。"""

    def __init__(
        self,
        tiling_cfg: Optional[TilingConfig] = None,
        ground_cfg: Optional[GroundFilterConfig] = None,
        dem_cfg: Optional[DEMGridConfig] = None,
        reader: Optional[ChunkReader] = None,
    ):
        self.tcfg = tiling_cfg or TilingConfig()
        self.gcfg = ground_cfg or GroundFilterConfig()
        self.dcfg = dem_cfg or DEMGridConfig()
        self.reader = reader or default_chunk_reader

    def process_file(self, path: str) -> TiledDEM:
        """对点云文件做完整分瓦片处理，返回整测区拼接近区 DEM。"""
        chunk = self.tcfg.max_points_in_memory
        bbox = scan_bbox(path, self.reader, chunk)
        tiles = make_tiles(bbox, self.tcfg)

        res = self.dcfg.resolution
        xmin, ymin, xmax, ymax = bbox
        # 整测区输出格网尺寸
        nx = max(1, int(np.ceil((xmax - xmin) / res)) + 1)
        ny = max(1, int(np.ceil((ymax - ymin) / res)) + 1)
        mosaic = np.full((ny, nx), self.dcfg.nodata, dtype=np.float64)

        gfilter = GroundFilter(self.gcfg)

        for tile in tiles:
            pts = collect_tile_points(path, self.reader, chunk, tile)
            if len(pts) == 0:
                continue
            # 地面分类（在带缓冲范围内做，减小边界效应）
            ground = gfilter.extract_ground(pts)
            if len(ground) < 3:
                continue
            # 建该瓦片（带缓冲）DEM
            tile_dem = generate_dem(ground, self.dcfg)
            # 把瓦片 DEM 的「核心区」写入整测区 mosaic（裁掉缓冲带）
            self._paste_core(mosaic, xmin, ymin, res, tile, tile_dem)

        return TiledDEM(
            dem=mosaic, origin_x=xmin, origin_y=ymin,
            cell_size=res, crs=self.dcfg.crs, nodata=self.dcfg.nodata,
        )

    def process_array(self, points: np.ndarray) -> TiledDEM:
        """对内存中的点云数组分瓦片处理（小数据/测试用），逻辑同 process_file。"""
        bbox = (
            float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()),
        )
        tiles = make_tiles(bbox, self.tcfg)
        res = self.dcfg.resolution
        xmin, ymin, xmax, ymax = bbox
        nx = max(1, int(np.ceil((xmax - xmin) / res)) + 1)
        ny = max(1, int(np.ceil((ymax - ymin) / res)) + 1)
        mosaic = np.full((ny, nx), self.dcfg.nodata, dtype=np.float64)
        gfilter = GroundFilter(self.gcfg)

        x, y = points[:, 0], points[:, 1]
        for tile in tiles:
            m = (x >= tile.bx0) & (x <= tile.bx1) & (y >= tile.by0) & (y <= tile.by1)
            pts = points[m]
            if len(pts) == 0:
                continue
            ground = gfilter.extract_ground(pts)
            if len(ground) < 3:
                continue
            tile_dem = generate_dem(ground, self.dcfg)
            self._paste_core(mosaic, xmin, ymin, res, tile, tile_dem)

        return TiledDEM(
            dem=mosaic, origin_x=xmin, origin_y=ymin,
            cell_size=res, crs=self.dcfg.crs, nodata=self.dcfg.nodata,
        )

    @staticmethod
    def _paste_core(
        mosaic: np.ndarray, mos_ox: float, mos_oy: float, res: float,
        tile: TileSpec, tile_dem: DEMResult,
    ) -> None:
        """把瓦片 DEM 的核心区（去缓冲）粘贴到整测区 mosaic 的对应位置。"""
        # 核心区在 mosaic 中的行列范围
        c0 = int(round((tile.x0 - mos_ox) / res))
        r0 = int(round((tile.y0 - mos_oy) / res))
        c1 = int(round((tile.x1 - mos_ox) / res))
        r1 = int(round((tile.y1 - mos_oy) / res))
        c0, r0 = max(0, c0), max(0, r0)
        c1 = min(mosaic.shape[1], c1 + 1)
        r1 = min(mosaic.shape[0], r1 + 1)
        if c1 <= c0 or r1 <= r0:
            return
        # 对核心区每个 mosaic 像元，取其在瓦片 DEM 中的对应值
        for r in range(r0, r1):
            y = mos_oy + r * res
            tr = int(round((y - tile_dem.origin_y) / tile_dem.cell_size))
            if tr < 0 or tr >= tile_dem.dem.shape[0]:
                continue
            for c in range(c0, c1):
                x = mos_ox + c * res
                tc = int(round((x - tile_dem.origin_x) / tile_dem.cell_size))
                if tc < 0 or tc >= tile_dem.dem.shape[1]:
                    continue
                val = tile_dem.dem[tr, tc]
                if val != tile_dem.nodata:
                    mosaic[r, c] = val

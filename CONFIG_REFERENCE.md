# Configuration Reference

This document explains the top-level configuration keys in `config/terrain_correction.yaml`.

Top-level keys
---------------
- `crs` - Coordinate reference information
  - `epsg` (int): EPSG code (example: 4547)
  - `crs_string` (str): full CRS string (example: "EPSG:4547")
  - `units` (str): `meter` or `degree`

- `physics` - Physical constants used for terrain correction
  - `density` (float): crust density in kg/m^3
  - `gravitational_constant` (float)
  - `mgal_scale` (float)

- `ground_filter` - Ground classification settings
  - `method` (str): `csf`, `pmf`, `smrf`, or `deep`
  - `csf` (mapping): cloth parameters (see file for defaults)
  - `pmf` (mapping)
  - `deep` (mapping): deep model settings (path, block size)

- `tiling` - Tiling / streaming settings for large point clouds
  - `enable` (bool)
  - `tile_size` (float)
  - `buffer` (float)
  - `max_points_in_memory` (int)
  - `n_workers` (int)

- `dem_from_pointcloud` - Parameters that map directly to DEMGridConfig used in code
  - `resolution` (float)
  - `method` (str): `idw`/`tin`/`kriging`
  - `idw_power` (float), `idw_k` (int), `idw_search_radius` (float)
  - `crs` (str), `nodata` (float)

- `dem_fusion` - DEM fusion settings
  - `target_resolution` (float)
  - `transition_width` (float)
  - `method` (str): `linear`/`cosine`/`sigmoid`
  - `sigmoid_k` (float)
  - `datum_align` (str): `none`/`median`/`plane`
  - `resample` (str)
  - `feather` (mapping) - nested copy for human readability

- `zone_integration` - Exact integration settings
  - `inner_radius`, `outer_radius`, `far_downsample`
  - `zone_depth` (float) and `rho` (float) are present for compatibility with train scripts

- `pstinet` - Neural inference settings
  - `enable`, `model_path`, `patch_size`, `in_channels`, `device`
  - `training` (mapping) - batch_size, num_epochs, learning_rate, weight_decay, patience, device
  - `losses` (mapping) - weights for loss components

- `recommender` - method recommendation engine settings
  - `mode` (str): `auto`/`recommend`/`both`
  - `enabled_methods` (mapping boolean)

- `output` - output preferences
  - `dir` (path): default results dir
  - `formats` (list): e.g. ["geotiff","npy","png","json"]
  - `png`/`geotiff` (mapping) for format-specific options

- `logging` and `performance` sections exist for advanced tuning.

Editing the config
------------------
- To change any default, copy `config/terrain_correction.yaml` to your project and modify the values.
- When running with `TerrainCorrector(config_file=...)`, the file will be loaded and used to configure submodules.

Compatibility note
------------------
- The project supports a limited set of legacy key names. If you have older config files, please run them through the normalize function or contact the maintainer to help migrate.

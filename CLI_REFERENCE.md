# CLI Reference

This document describes the command-line interface for the `terrain-correction` tool.

Install
-------
After installing the package in editable mode:

```bash
pip install -e .
```

Available commands
------------------
- `terrain-correction` - top-level helper. Subcommands:

1) `terrain-correction serve`
   - Start a lightweight web UI (Flask) for uploading point clouds, launching jobs and downloading results.
   - Example:
     ```bash
     terrain-correction serve --host 0.0.0.0 --port 5000
     ```

2) `terrain-correction process`
   - Run processing pipeline in CLI mode (non-interactive).
   - Example:
     ```bash
     terrain-correction process --points data/sample_points.npy --output ./results
     ```
   - Options:
     --points <path>        Path to point cloud (.npy/.xyz/.las)
     --dem_far <path>       Optional far-zone DEM to blend
     --config <path>        Path to YAML config (default: config/terrain_correction.yaml)
     --basename <name>      Output basename

3) `terrain-correction export`
   - Export an existing correction grid with configured formats.
   - Example:
     ```bash
     terrain-correction export --grid ./tmp/correction.npy --meta ./tmp/meta.json --out ./results
     ```

Notes
-----
- The CLI is convenient for batch jobs and remote execution (no browser required).
- Use `terrain-correction --help` or `terrain-correction <subcommand> --help` for detailed options.

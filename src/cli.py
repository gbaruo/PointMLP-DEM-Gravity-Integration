"""CLI entry point for the package.

Provides subcommands:
 - serve: run the lightweight Flask web UI
 - process: run pipeline from CLI
 - export: helper to export existing grid (placeholder)

This is a thin wrapper; heavy lifting is performed by src.terrain_correction.TerrainCorrector
"""

import argparse
import sys
from pathlib import Path


def cmd_serve(args):
    # Lazy import to avoid requiring Flask on simple CLI usage
    try:
        from src.webapp.app import create_app
    except Exception as e:
        print("Unable to import webapp: make sure Flask is installed (pip install flask)")
        raise

    app = create_app()
    host = args.host or "127.0.0.1"
    port = int(args.port or 5000)
    print(f"Starting web UI on http://{host}:{port}")
    app.run(host=host, port=port)


def cmd_process(args):
    try:
        from src.terrain_correction import TerrainCorrector
    except Exception as e:
        print("Failed importing TerrainCorrector:", e)
        sys.exit(1)

    points = args.points
    output = args.output or "./results"
    config = args.config
    dem_far = args.dem_far
    basename = args.basename or "correction"

    if not points:
        print("--points is required for process command")
        sys.exit(1)

    print("Loading TerrainCorrector...")
    if config:
        tc = TerrainCorrector(config_file=str(config))
    else:
        # Lightweight automatic configuration if config not supplied
        tc = TerrainCorrector.auto_configure(n_points=100_000, precision="cm")

    print(f"Processing points: {points} -> output: {output}")
    try:
        res = tc.process_and_export(points_file=str(points), dem_far_file=(str(dem_far) if dem_far else None), output_dir=str(output), basename=basename)
        print("Processing finished. Outputs:")
        for k, v in (res or {}).items():
            print(f"  {k}: {v}")
    except Exception as e:
        print("Processing failed:", e)
        raise


def cmd_export(args):
    # Simple helper - if you have a saved numpy grid + meta, use src.output.export_result
    try:
        from src.output import export_result
        import numpy as np
    except Exception:
        print("export requires src.output and numpy to be available")
        sys.exit(1)

    grid = args.grid
    meta = args.meta
    out = args.out or "./results"
    basename = args.basename or "exported"

    if not grid or not meta:
        print("--grid and --meta are required")
        sys.exit(1)

    grid = Path(grid)
    meta = Path(meta)
    if not grid.exists() or not meta.exists():
        print("grid or meta file not found")
        sys.exit(1)

    arr = np.load(str(grid))
    import json
    m = json.load(open(str(meta)))

    export_result(arr, m.get("origin_x"), m.get("origin_y"), m.get("cell_size"), str(out), basename, crs=m.get("crs"), formats=["npy","tif","png"], description="exported via CLI")
    print("Export completed")


def main():
    parser = argparse.ArgumentParser(prog="terrain-correction")
    sub = parser.add_subparsers(dest="cmd")

    p_serve = sub.add_parser("serve", help="Run web UI")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", default=None)
    p_serve.set_defaults(func=cmd_serve)

    p_proc = sub.add_parser("process", help="Run processing pipeline in CLI")
    p_proc.add_argument("--points", required=True)
    p_proc.add_argument("--dem_far", required=False)
    p_proc.add_argument("--config", required=False)
    p_proc.add_argument("--output", required=False)
    p_proc.add_argument("--basename", required=False)
    p_proc.set_defaults(func=cmd_process)

    p_exp = sub.add_parser("export", help="Export an existing grid with metadata")
    p_exp.add_argument("--grid", required=True)
    p_exp.add_argument("--meta", required=True)
    p_exp.add_argument("--out", required=False)
    p_exp.add_argument("--basename", required=False)
    p_exp.set_defaults(func=cmd_export)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

"""Small deterministic synthetic sample generator for tests and demos.

Generates a small point cloud with a base plane + sinusoidal terrain + random vegetation points.
Usage:
    python data/generate_sample.py --out data/sample_points.npy
"""

import argparse
import numpy as np
from pathlib import Path


def generate(n_ground=5000, n_veg=1000, seed=42):
    np.random.seed(seed)
    x_g = np.random.uniform(0, 500, n_ground)
    y_g = np.random.uniform(0, 500, n_ground)
    z_g = 50 + 0.02 * x_g + 0.01 * y_g + 2.0 * np.sin(0.01 * x_g) + np.random.normal(0, 0.1, n_ground)
    ground = np.column_stack([x_g, y_g, z_g])

    x_v = np.random.uniform(0, 500, n_veg)
    y_v = np.random.uniform(0, 500, n_veg)
    z_v = 50 + 0.02 * x_v + 0.01 * y_v + np.random.uniform(2.0, 15.0, n_veg)
    veg = np.column_stack([x_v, y_v, z_v])

    pts = np.vstack([ground, veg])
    return pts


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/sample_points.npy')
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pts = generate()
    np.save(out, pts.astype(np.float32))
    print(f"Saved sample points to {out}")

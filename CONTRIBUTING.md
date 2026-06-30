# Contributing

Thank you for wanting to contribute to PointMLP-DEM-Gravity-Integration! This document explains how to contribute code, file issues, and submit pull requests so we can collaborate smoothly.

1. Code of Conduct
   - Follow respectful behavior. See CODE_OF_CONDUCT.md (if present).

2. Development Workflow
   - Fork the repository and create a topic branch from `feat/adaptive-terrain-correction` (or `main` if you're working on releases).
   - Branch name convention: `feat/<short-descriptor>`, `fix/<short-descriptor>`, `docs/<short>`.

3. Set up a development environment
   - Create and activate virtualenv or conda env:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     pip install -e .[dev]
     ```
   - Alternatively with conda:
     ```bash
     conda create -n pmlp python=3.9
     conda activate pmlp
     pip install -e .[dev]
     ```

4. Formatting & Linting
   - We use `black`, `isort`, and `flake8`.
   - Run formatting locally before committing:
     ```bash
     black .
     isort .
     flake8
     ```

5. Tests
   - Write tests under `tests/` using `pytest`.
   - Run tests and coverage locally:
     ```bash
     pytest -q
     ```

6. Commits & Pull Requests
   - Keep commits focused and atomic.
   - Use descriptive commit messages. Example:
     ```
     feat(dem): add bilinear resampling option to dem_fusion
     ```
   - Open a Pull Request to `feat/adaptive-terrain-correction` for feature work or to `main` for release patches.
   - Include tests and update documentation when adding features.

7. Review
   - PRs will be reviewed; respond to review comments and push updates to the same branch.

8. Security
   - For security issues, do not open a public issue. Contact the maintainer at the address in `README.md` (or email).

---

If you want, add your name and GitHub handle to `CONTRIBUTORS.md` after your first merged PR.

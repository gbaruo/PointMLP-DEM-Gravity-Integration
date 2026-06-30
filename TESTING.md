# Testing Guide

This project uses `pytest` for unit and integration tests. The tests are intended to be fast, deterministic, and to exercise core functionality on CPU.

Quickstart
----------
1. Create a development environment and install dev dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

2. Run the full test suite:

```bash
pytest -q
```

3. Run a single test file:

```bash
pytest tests/test_dem_from_pointcloud.py -q
```

GPU tests
---------
GPU-dependent tests will be marked with `@pytest.mark.gpu`. By default CI does not run GPU tests. Locally you can run them with `-m gpu` if you have CUDA available and `torch` installed.

Coverage
--------
To run tests with coverage and generate an HTML report:

```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html
```

Writing tests
-------------
- Tests live in `tests/` and should use `pytest` fixtures when appropriate.
- Keep tests small and deterministic; avoid network or external filesystem dependencies.
- Use the small synthetic data generator `data/generate_sample.py` for deterministic examples.

Failing tests
-------------
- If a test fails on CI but passes locally, check for environment differences (Python version, optional deps).
- Mark slow external tests with `@pytest.mark.slow` and GPU tests with `@pytest.mark.gpu`.

Tips
----
- Use `pytest -k <expr>` to run tests matching expr.
- Use `pytest -x` to stop on first failure.

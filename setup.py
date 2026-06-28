"""setup.py: 包安装配置。

支持: pip install -e .  (开发模式)
      pip install .      (生产模式)
      python setup.py sdist bdist_wheel  (打包分发)
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README 作长描述
here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# 读取 requirements.txt
requirements = (here / "requirements.txt").read_text(encoding="utf-8").strip().split("\n")
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="PointMLP-DEM-Gravity-Integration",
    version="0.1.0",
    description="Physics-guided Adaptive Terrain Correction via Point Cloud DEM Fusion and Neural Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="gbaruo",
    author_email="your.email@example.com",  # 改为你的邮箱
    url="https://github.com/gbaruo/PointMLP-DEM-Gravity-Integration",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="terrain-correction gravity DEM point-cloud neural-network",
    packages=find_packages(where=".", include=["src", "src.*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "torch>=1.12",
        ],
        "geo": [
            "rasterio>=1.2",
            "fiona>=1.8",
        ],
        "viz": [
            "matplotlib>=3.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "terrain-correction=src.terrain_correction:main",  # 命令行入口(可选)
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/gbaruo/PointMLP-DEM-Gravity-Integration/issues",
        "Source": "https://github.com/gbaruo/PointMLP-DEM-Gravity-Integration",
        "Documentation": "https://github.com/gbaruo/PointMLP-DEM-Gravity-Integration/blob/main/USAGE.md",
    },
)

"""
إعداد حزمة نظام GSE المتقدم
Setup script for Advanced GSE System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gse-advanced-system",
    version="1.0.0",
    author="مبتكر المعادلات التكيفية",
    author_email="developer@gse-system.com",
    description="نظام المعادلة السيجمويدية المعممة المتقدم لتقريب الدوال المعقدة",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gse-advanced-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "interactive": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gse-run=src.main_advanced:main",
            "gse-demo=src.main_advanced:demo_interactive",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

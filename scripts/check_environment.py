"""Dependency/environment check: verifies every runtime import works.

Usage:
    python scripts/check_environment.py
"""
from __future__ import annotations

import importlib
import sys

REQUIRED = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("streamlit", "streamlit"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("joblib", "joblib"),
    ("dotenv", "python-dotenv"),
    ("pytest", "pytest"),
    ("httpx", "httpx"),
    ("httpx2", "httpx2"),
]


def main() -> int:
    print("Environment check")
    print("-----------------")
    print(f"Python: {sys.version.split()[0]}")
    missing: list[str] = []
    for module, package in REQUIRED:
        try:
            mod = importlib.import_module(module)
            print(f"{package}: OK ({getattr(mod, '__version__', 'installed')})")
        except ImportError:
            print(f"{package}: MISSING")
            missing.append(package)
    if missing:
        print("\nInstall missing packages with:")
        print("    python -m pip install -r requirements.txt")
        return 1
    print("\nEnvironment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

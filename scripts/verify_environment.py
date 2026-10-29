"""Print versions of every dependency, GPU availability, and Earth Engine auth status.

Run:  python scripts/verify_environment.py
"""
import importlib
import platform
import shutil
import sys
from pathlib import Path

PACKAGES = ["numpy", "pandas", "torch", "rasterio", "geopandas", "shapely", "pyproj", "pyogrio", "cv2",
            "sklearn", "scipy", "matplotlib", "yaml", "ee", "pystac_client", "planetary_computer", "s2sphere",
            "pytest"]


def main():
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print(f"Interpreter: {sys.executable}")
    ok = True
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  OK   {name:20s} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # report every missing package, then fail
            ok = False
            print(f"  FAIL {name:20s} {exc}")
    try:
        import rasterio
        print(f"GDAL (bundled with rasterio): {rasterio.__gdal_version__}")
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}",
              f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
    except Exception:
        pass
    total, used, free = shutil.disk_usage(Path(__file__).resolve().parents[1])
    print(f"Disk free: {free / 1e9:.1f} GB")
    cred = Path.home() / ".config" / "earthengine" / "credentials"
    print(f"Earth Engine credentials present: {cred.exists()}"
          + ("" if cred.exists() else "  -> run .\\.venv\\Scripts\\earthengine.exe authenticate"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Record the machine that produced every number → reports/env.json.

PLAN.md §5.1: "Every compute claim in the report cites this file."  Tier A runs
anywhere; Tier B needs CUDA, so this also answers the only question that matters
before committing six hours to a pretraining run.

    python -m bnqa.envinfo
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys

from .config import CFG, ENV_JSON, ROOT, config_hash, ensure_dirs
from .utils import write_json


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _torch_info() -> dict:
    """Tier B probe.  Absent torch is a fact about the machine, not an error."""
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda_available": False,
                "note": "torch not installed - Tier A only on this machine"}

    info: dict = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        info.update(
            gpu_name=props.name,
            gpu_total_vram_gb=round(props.total_memory / 1024**3, 2),
            gpu_free_vram_gb=round(free / 1024**3, 2),
            gpu_capability=f"{props.major}.{props.minor}",
            bf16_supported=bool(torch.cuda.is_bf16_supported()),
        )
    return info


def collect() -> dict:
    import numpy
    import scipy
    import sklearn

    total, used, free = shutil.disk_usage(ROOT)
    env = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "ram_gb": _ram_gb(),
        "disk_free_gb": round(free / 1024**3, 1),
        "root": str(ROOT),
        "git_commit": _git_commit(),
        "config_hash": config_hash(),
        "seed": CFG.seed,
        "packages": {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": __import__("pandas").__version__,
            "pyarrow": __import__("pyarrow").__version__,
        },
        "tier_b": _torch_info(),
    }
    env["tier_a_ready"] = True
    env["tier_b_ready"] = bool(env["tier_b"].get("cuda_available"))
    return env


def _ram_gb() -> float | None:
    if sys.platform == "win32":  # GlobalMemoryStatusEx — no psutil, no wmic
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        try:
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            return round(st.ullTotalPhys / 1024**3, 1)
        except Exception:
            return None
    try:  # POSIX
        import os

        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except Exception:
        return None


def main() -> None:
    ensure_dirs()
    env = collect()
    write_json(ENV_JSON, env)
    print(f"wrote {ENV_JSON}")
    print(f"  python      {env['python']}  ({env['cpu_count']} cpus, {env['ram_gb']} GB RAM)")
    print(f"  disk free   {env['disk_free_gb']} GB")
    print(f"  config hash {env['config_hash']}")
    print(f"  Tier A      ready")
    if env["tier_b_ready"]:
        b = env["tier_b"]
        print(f"  Tier B      READY - {b['gpu_name']}, {b['gpu_total_vram_gb']} GB VRAM, "
              f"bf16={b['bf16_supported']}")
    else:
        print(f"  Tier B      not available here ({env['tier_b'].get('note') or 'no CUDA device'})")


if __name__ == "__main__":
    main()

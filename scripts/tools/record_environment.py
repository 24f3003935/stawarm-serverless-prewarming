"""Record the software and hardware environment of a run, so that results can be compared across machines.

Writes a text file with the Python version, operating system, CPU and memory, GPU (if any), the versions of
the packages the experiments depend on, and the full list of installed packages (pip freeze).

Run : python scripts/tools/record_environment.py results/environment.txt
"""
from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch", "matplotlib", "gdown"]


def memory_gb() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])
            return f"{kb / 1024 ** 2:.1f} GB"
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return f"{m.ullTotalPhys / 1024 ** 3:.1f} GB"
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
            return f"{int(out) / 1024 ** 3:.1f} GB"
    except Exception:
        pass
    return "unknown"


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/environment.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"python      {sys.version.split()[0]} ({platform.python_implementation()})",
             f"platform    {platform.platform()}",
             f"processor   {platform.processor() or platform.machine()}",
             f"cpu cores   {os.cpu_count()}",
             f"memory      {memory_gb()}"]
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            lines.append(f"{name:11s} {getattr(mod, '__version__', 'unknown')}")
        except Exception:
            lines.append(f"{name:11s} not installed")
    try:
        import torch
        if torch.cuda.is_available():
            lines.append(f"gpu         {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}, "
                         f"{torch.cuda.device_count()} device(s))")
        else:
            lines.append("gpu         none (CUDA not available)")
        lines.append(f"torch threads {torch.get_num_threads()}")
    except Exception:
        pass
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    text = "\n".join(lines) + "\n\n# pip freeze\n" + freeze
    out.write_text(text, encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

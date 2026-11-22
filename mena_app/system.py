"""Machine facts that need no torch: memory, load, cores. Used by the server to protect the computer from big tests."""

from __future__ import annotations

import os
import platform
import subprocess
import sys


def _sysctl(name: str) -> int | None:
    try:
        return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=3).stdout.strip())
    except Exception:
        return None


def _win_mem_status() -> tuple[int, int] | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MEMORYSTATUSEX(dwLength=ctypes.sizeof(_MEMORYSTATUSEX))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullTotalPhys), int(st.ullAvailPhys)
    except Exception:
        pass
    return None


def process_rss_bytes() -> int:
    """Peak resident set size of the current process in bytes (Unix/macOS via `resource`, Windows via Win32 API)."""
    try:
        import resource

        ru = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return ru if sys.platform == "darwin" else ru * 1024
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = _PROCESS_MEMORY_COUNTERS(cb=ctypes.sizeof(_PROCESS_MEMORY_COUNTERS))
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p  # a pseudo-handle: must not be truncated to a 32-bit int
            fn = getattr(k32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
            fn.restype = ctypes.c_int
            if fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return int(pmc.PeakWorkingSetSize)
        except Exception:
            pass
    return 0


def total_memory_bytes() -> int:
    if sys.platform == "darwin":
        v = _sysctl("hw.memsize")
        if v:
            return v
    win = _win_mem_status()
    if win is not None:
        return win[0]
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def available_memory_bytes() -> int:
    """Memory that can be handed out without swapping: free + inactive + speculative + purgeable pages on macOS,
    MemAvailable on Linux, ullAvailPhys on Windows. Returns 0 when it cannot be read (callers then skip the guard and say so)."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3).stdout
            page = 4096
            pages: dict[str, int] = {}
            for line in out.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split()[0])
                elif ":" in line:
                    k, v = line.split(":", 1)
                    v = v.strip().rstrip(".")
                    if v.isdigit():
                        pages[k.strip()] = int(v)
            n = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
            return n * page
        win = _win_mem_status()
        if win is not None:
            return win[1]
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def performance_cores() -> int:
    if sys.platform == "darwin":
        v = _sysctl("hw.perflevel0.logicalcpu")
        if v:
            return v
    return max(1, os.cpu_count() or 1)


def cuda_vram_info() -> list[dict]:
    """Query per-GPU free/total VRAM (GB) via nvidia-smi without importing torch or allocating a CUDA context."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        r = subprocess.run(
            [exe, "--query-gpu=index,name,memory.free,memory.total,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return []
        rows: list[dict] = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[0])
                free_mib = float(parts[2])
                tot_mib = float(parts[3])
            except ValueError:
                continue
            row = {
                "index": idx,
                "id": f"cuda:{idx}",
                "name": parts[1],
                "vram_free_bytes": int(free_mib * 1024 * 1024),
                "vram_total_bytes": int(tot_mib * 1024 * 1024),
                "vram_free_gb": round(free_mib * 1024 * 1024 / 1e9, 2),
                "vram_total_gb": round(tot_mib * 1024 * 1024 / 1e9, 2),
            }
            if len(parts) >= 5 and parts[4]:
                row["uuid"] = parts[4]
            rows.append(row)
        return rows
    except Exception:
        return []


def _visible_cuda_tokens() -> list[str] | None:
    """Return physical CUDA selectors exposed via CUDA_VISIBLE_DEVICES, or None if unset."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    s = raw.strip()
    if s == "" or s.lower() in ("-1", "none", "void", "nodevfiles"):
        return []
    vis: list[str] = []
    for tok in s.split(","):
        t = tok.strip()
        if not t:
            continue
        if t.startswith("-") and t[1:].isdigit():
            break
        vis.append(t)
    return vis


def cuda_row_for_logical_index(rows: list[dict], device_index: int = 0) -> dict | None:
    """Resolve logical `cuda:{device_index}` against physical `nvidia-smi` rows honoring `CUDA_VISIBLE_DEVICES`."""
    idx = int(device_index)
    vis = _visible_cuda_tokens()
    if vis is None:
        for row in rows:
            if row.get("index") == idx:
                return row
        return None
    if idx < 0 or idx >= len(vis):
        return None
    sel = vis[idx]
    if sel.isdigit():
        phys = int(sel)
        for row in rows:
            if row.get("index") == phys:
                return row
        return None
    for row in rows:
        uuid = str(row.get("uuid", ""))
        if uuid and (uuid == sel or uuid.startswith(sel)):
            return row
    return None


def available_vram_bytes(device_index: int = 0) -> int | None:
    """Live free VRAM in bytes for logical `cuda:{device_index}` without creating a CUDA context, or None if unavailable."""
    row = cuda_row_for_logical_index(cuda_vram_info(), device_index)
    return None if row is None else int(row["vram_free_bytes"])


def snapshot() -> dict:
    total, avail = total_memory_bytes(), available_memory_bytes()
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):
        load = None
    cores = os.cpu_count() or 1
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "total_gb": round(total / 1e9, 2), "available_gb": round(avail / 1e9, 2) if avail else None,
        "cpu_count": cores, "performance_cores": performance_cores(),
        "load_1min": None if load is None else round(load, 2),
        "busy": bool(load is not None and load > 0.7 * cores),
    }

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

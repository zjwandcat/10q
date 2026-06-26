"""
M5 Optimizer Professional Launcher
===================================
Pre-flight checks before starting the Gradio server.
Called by:  py -m m5_optimizer._launcher

Checks:
  1. Python version
  2. Duplicate instance
  3. Port availability
  4. Dependencies (smart install)
  5. Config integrity
  6. Data directories
  7. Disk space
  8. Memory status
  9. GPU availability (optional)
  10. Old log cleanup
"""
import os
import sys
import gc
import time
import socket
import shutil
import subprocess
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ──
REQUIRED_MODULES = {
    "optuna": "optuna",
    "gradio": "gradio",
    "yaml": "pyyaml",
    "psutil": "psutil",
    "numpy": "numpy",
    "pandas": "pandas",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
}
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
BASE_PORT = 7860
MAX_PORT = 7869
PIP_CHECK_FILE = os.path.join("logs", "m5", ".pip_check")
MIN_DISK_GB = 1.0  # minimum free disk space in GB
LOG_RETAIN_DAYS = 30  # auto-cleanup logs older than this

# ── ANSI colors (Windows 10+ cmd supports them) ──
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _p(msg: str = ""):
    print(msg, flush=True)


def _ok(msg: str):
    _p(f"  {_GREEN}OK{_RESET}  {msg}")


def _warn(msg: str):
    _p(f"  {_YELLOW}WARN{_RESET} {msg}")


def _fail(msg: str):
    _p(f"  {_RED}FAIL{_RESET} {msg}")


def _info(msg: str):
    _p(f"  {_CYAN}INFO{_RESET} {msg}")


def _step(n: int, total: int, label: str):
    _p(f"\n  {_DIM}[{n}/{total}]{_RESET} {label}")


def _bar():
    _p("  " + "-" * 52)


# ═══════════════════════════════════════════════════════════
#  Check 1: Python version
# ═══════════════════════════════════════════════════════════
def check_python():
    v = sys.version_info
    _p(f"  Python {v.major}.{v.minor}.{v.micro}  ({sys.executable})")
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        _fail("Python 3.9+ required!")
        return False
    _ok(f"Python {v.major}.{v.minor} meets requirement (>=3.9)")
    return True


# ═══════════════════════════════════════════════════════════
#  Check 2: Duplicate instance
# ═══════════════════════════════════════════════════════════
def check_duplicate():
    try:
        import psutil
        current_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["pid"] == current_pid:
                    continue
                if not proc.info["name"] or "python" not in proc.info["name"].lower():
                    continue
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if "m5_optimizer.app" in cmdline and "_launcher" not in cmdline:
                    _warn("M5 optimizer is already running!")
                    _info("Close the existing window to avoid port conflict.")
                    return True  # not fatal
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        _info("psutil not available, skipping duplicate check.")
        return True
    _ok("No duplicate instance detected.")
    return True


# ═══════════════════════════════════════════════════════════
#  Check 3: Port availability
# ═══════════════════════════════════════════════════════════
def check_port():
    available = []
    for port in range(BASE_PORT, MAX_PORT + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                available.append(port)
            except OSError:
                pass
    if available:
        _ok(f"Port {available[0]} available (range {available[0]}-{available[-1]})")
    else:
        _warn("Ports 7860-7869 all in use! App will try auto-fallback.")
    return available


# ═══════════════════════════════════════════════════════════
#  Check 4: Dependencies
# ═══════════════════════════════════════════════════════════
def has_network():
    try:
        with socket.create_connection(("223.5.5.5", 53), timeout=2):
            return True
    except OSError:
        return False


def check_dependencies():
    missing = []
    for import_name, pip_name in REQUIRED_MODULES.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append((import_name, pip_name))

    if not missing:
        _ok("All core dependencies installed.")
        return True

    for import_name, pip_name in missing:
        _warn(f"Missing: {import_name} (pip install {pip_name})")

    net = has_network()
    if not net:
        _warn("No network - skipping install. App may fail if deps are truly missing.")
        _info("Connect to network and delete logs/m5/.pip_check to retry.")
        return True

    pip_names = list(dict.fromkeys(pip_name for _, pip_name in missing))
    _info(f"Installing: {', '.join(pip_names)} ...")

    # Try mirror first, then default, no more retries
    cmd = [sys.executable, "-m", "pip", "install"] + pip_names + [
        "-i", PIP_MIRROR, "--upgrade", "--quiet"
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        _info("Mirror failed, trying default source...")
        cmd = [sys.executable, "-m", "pip", "install"] + pip_names + [
            "--upgrade", "--quiet"
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True)

    if ret.returncode != 0:
        _fail(f"Install failed: {ret.stderr.strip()[:200]}")
        return False

    _ok("Dependencies installed successfully!")
    return True


def upgrade_dependencies():
    """Silent upgrade if check mark not present."""
    if os.path.isfile(PIP_CHECK_FILE):
        _info("Pip check mark exists, skipping upgrade.")
        return

    if not has_network():
        _info("No network, skipping upgrade (will retry when online).")
        return

    _info("Upgrading dependencies (silent)...")
    pip_names = list(dict.fromkeys(REQUIRED_MODULES.values()))
    cmd = [sys.executable, "-m", "pip", "install"] + pip_names + [
        "-i", PIP_MIRROR, "--upgrade", "--quiet"
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        cmd = [sys.executable, "-m", "pip", "install"] + pip_names + [
            "--upgrade", "--quiet"
        ]
        subprocess.run(cmd, capture_output=True, text=True)

    Path(PIP_CHECK_FILE).write_text("checked")
    _ok("Dependencies updated.")


# ═══════════════════════════════════════════════════════════
#  Check 5: Config integrity
# ═══════════════════════════════════════════════════════════
def check_config():
    ok = True
    config_path = Path("config/config.yaml")
    if not config_path.is_file():
        _fail("config/config.yaml not found!")
        ok = False
    else:
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                _fail("config.yaml is not a valid YAML dict!")
                ok = False
            else:
                # Check key sections
                for key in ("data", "m5"):
                    if key not in cfg:
                        _warn(f"config.yaml missing section: {key}")
                    else:
                        pass
                _ok("config.yaml loaded and valid.")
        except Exception as e:
            _fail(f"config.yaml parse error: {e}")
            ok = False

    app_path = Path("m5_optimizer/app.py")
    if not app_path.is_file():
        _fail("m5_optimizer/app.py not found!")
        ok = False
    else:
        _ok("app.py exists.")

    return ok


# ═══════════════════════════════════════════════════════════
#  Check 6: Data directories
# ═══════════════════════════════════════════════════════════
def check_data_dirs():
    """Verify data directories referenced in config exist."""
    ok = True
    try:
        import yaml
        with open("config/config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        pool_dirs = cfg.get("data", {}).get("pool_dirs", {})
        active = cfg.get("data", {}).get("neutralization", {}).get("active_scheme", "")
        if active and active in pool_dirs:
            p = Path(pool_dirs[active])
            if p.is_dir():
                n_files = len(list(p.glob("*.parquet")))
                _ok(f"Data pool: {p} ({n_files} parquet files)")
            else:
                _warn(f"Data pool not found: {p}")
                ok = False
        else:
            _info("No active data pool configured.")

        index_dir = cfg.get("data", {}).get("index_dir", "")
        if index_dir and Path(index_dir).is_dir():
            _ok(f"Index data: {index_dir}")
        elif index_dir:
            _warn(f"Index data dir not found: {index_dir}")

    except Exception as e:
        _warn(f"Could not verify data dirs: {e}")

    return ok


# ═══════════════════════════════════════════════════════════
#  Check 7: Disk space
# ═══════════════════════════════════════════════════════════
def check_disk_space():
    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_pct = usage.used / usage.total * 100

        if free_gb < MIN_DISK_GB:
            _fail(f"Disk space low: {free_gb:.1f} GB free (need {MIN_DISK_GB} GB)")
            return False
        _ok(f"Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total ({used_pct:.0f}% used)")
        return True
    except Exception as e:
        _warn(f"Could not check disk space: {e}")
        return True


# ═══════════════════════════════════════════════════════════
#  Check 8: Memory status
# ═══════════════════════════════════════════════════════════
def check_memory():
    try:
        import psutil
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)

        if avail_gb < 2.0:
            _warn(f"Memory low: {avail_gb:.1f} GB available / {total_gb:.1f} GB total")
        else:
            _ok(f"Memory: {avail_gb:.1f} GB available / {total_gb:.1f} GB total")
    except ImportError:
        _info("psutil not available, skipping memory check.")
    return True


# ═══════════════════════════════════════════════════════════
#  Check 9: GPU availability (optional, info only)
# ═══════════════════════════════════════════════════════════
def check_gpu():
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            _info(result.stdout.strip())
        else:
            _info("No CUDA GPU detected (CPU mode).")
    except Exception:
        _info("PyTorch not installed, GPU check skipped.")
    return True


# ═══════════════════════════════════════════════════════════
#  Check 10: Old log cleanup
# ═══════════════════════════════════════════════════════════
def cleanup_old_logs():
    log_dir = Path("logs/m5")
    if not log_dir.is_dir():
        return True

    cutoff = datetime.now() - timedelta(days=LOG_RETAIN_DAYS)
    cleaned = 0
    for f in log_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff.timestamp():
            # Don't delete .pip_check or diag.log
            if f.name in (".pip_check", "diag.log", "crash.log"):
                continue
            try:
                f.unlink()
                cleaned += 1
            except OSError:
                pass

    if cleaned > 0:
        _ok(f"Cleaned {cleaned} old log file(s) (>{LOG_RETAIN_DAYS}d).")
    else:
        _ok("No old logs to clean.")
    return True


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    _p()
    _p("  ========================================================")
    _p("        TTHH M5 Bayesian Hyperparameter Optimizer")
    _p("  ========================================================")
    _p()

    TOTAL = 10
    fatal = False

    # 1. Python
    _step(1, TOTAL, "Python version")
    if not check_python():
        fatal = True

    # 2. Duplicate
    _step(2, TOTAL, "Duplicate instance")
    check_duplicate()

    # 3. Port
    _step(3, TOTAL, "Port availability")
    check_port()

    # 4. Dependencies
    _step(4, TOTAL, "Dependencies")
    if not check_dependencies():
        fatal = True
    else:
        upgrade_dependencies()

    # 5. Config
    _step(5, TOTAL, "Config integrity")
    if not check_config():
        fatal = True

    # 6. Data dirs
    _step(6, TOTAL, "Data directories")
    check_data_dirs()

    # 7. Disk
    _step(7, TOTAL, "Disk space")
    check_disk_space()

    # 8. Memory
    _step(8, TOTAL, "Memory status")
    check_memory()

    # 9. GPU
    _step(9, TOTAL, "GPU availability")
    check_gpu()

    # 10. Log cleanup
    _step(10, TOTAL, "Log cleanup")
    cleanup_old_logs()

    _p()
    _bar()

    elapsed = time.time() - t0
    _p(f"  Pre-flight checks completed in {elapsed:.1f}s")

    if fatal:
        _p()
        _fail("Fatal errors detected, cannot start.")
        _p("  Fix the errors above and try again.")
        _p("  To force pip upgrade: delete logs/m5/.pip_check")
        _p()
        input("  Press Enter to exit...")
        sys.exit(1)

    _p()
    _p("  Starting Gradio server...")
    _bar()
    _p()

    # ── Launch the app ──
    import yaml
    from m5_optimizer.app import build_app

    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    host = cfg.get("m5", {}).get("gradio", {}).get("server_name", "0.0.0.0")
    base_port = int(
        cfg.get("m5", {}).get("gradio", {}).get("server_port", 7860)
    )

    app = build_app()

    # Port fallback
    for offset in range(10):
        port = base_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        try:
            app.launch(
                server_name=host,
                server_port=port,
                share=False,
                inbrowser=(offset == 0),
            )
            return
        except OSError:
            continue

    _fail("No available port found!")
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _p("\n  M5 Optimizer stopped by user.")
    except Exception as e:
        _p(f"\n  [FATAL] {e}")
        import traceback
        traceback.print_exc()
        os.makedirs("logs/m5", exist_ok=True)
        with open("logs/m5/crash.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {e}\n{traceback.format_exc()}\n")
        input("\n  Press Enter to exit...")
        sys.exit(1)

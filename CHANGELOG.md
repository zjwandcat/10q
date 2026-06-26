# Changelog

All notable changes to TTHH will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [4.2] - 2026-06-17

### Fixed
- **Trial hang 6h → OOM** (root cause: GPU `inplace_predict` deadlock).
  Added 60s timeout on XGB GPU predict, 300s on parallel training, 1800s on whole Trial.
- **Phase1 crash on `KeyError: Record does not exist`** during cleanup.
  `trial.set_user_attr` now swallows `KeyError`; `cleanup_bad_trials` accepts `skip_running`.
- **`updated_state` UnboundLocalError** propagation to study loop.
  Stop button passes `skip_running=True` so running Trials are not deleted from DB.
- **Memory leak (RSS growth) without stop**.
  `restart_check.check_rss_leak` triggers stop when RSS +20% and +0.5GB over 20 trials.
- **Watchdog spamming CRITICAL every 15s at night**.
  Layered thresholds: warn <0.5GB, stop <0.2GB.
- **RSS not returned to OS after each Trial**.
  `gc.collect(2) + msvcrt.heapmin()` runs in trial callback.
- **Extreme M5 params (n_est=10000, depth=20, lr=0.0001) → 90h Trial**.
  Hard caps in `assemble_params`: n_est≤500, depth≤8, lr≥0.005.
- **Window loop ignores `stop_event` inside long window**.
  `m2` and `m2_gpu` `_process_single_window` check `stop_event` at 3 points.

### Added
- `tests/verify_v42_fixes.py` — 9 automated tests for the above (run via `.githooks/pre-commit`).

## [4.1] and earlier

See git history.

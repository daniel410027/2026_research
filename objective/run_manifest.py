"""
run_manifest.py
===============
把「這份產出是被什麼程式碼、什麼設定跑出來的」寫進產出目錄本身。

★ 2026-07-27 新增。動機（summary_research.md §6.1 的後遺症）：
  `_should_tune` 的凍結機制曾經失效——宣稱「用固定超參數跑」的產出，實際上
  每個 fold 都重跑了 Optuna。修好之後留下一個沒辦法回答的問題：
  **既有的 `database/experiment/` 哪些結果受影響？** 因為產出裡沒有任何關於
  「它是被哪一版程式碼、哪一組設定產生的」的記錄，只能全部視為不可信。

  對一個「產出會被下游實驗當成固定上游反覆使用」的 repo（§0.2），這是架構級
  缺口：實驗之間靠資料夾複製隔離（§0.1），同一個 `database/experiment/` 路徑
  在不同副本裡意義不同，沒有 manifest 就無法事後辨識。

  有了 manifest，下次再發現某個旗標語意不符，可以直接篩出受影響的舊產出
  （`code_hash` 或 `config.tune_mode` 一比就知道），而不是全部作廢重跑。

設計原則：**寫 manifest 絕不可以弄掉一次跑了 40 分鐘的訓練。** 每個區段都各自
try/except，失敗就在該欄位留下 {"error": ...}，不往外拋。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────
#  程式碼指紋
# ─────────────────────────────────────────────────────────────

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_fingerprint(pkg_dir: Path = _PKG_DIR) -> dict:
    """
    `objective/` 套件全部 .py 的內容指紋。

    per-file hash 讓「哪一支檔案變了」一眼可見；combined hash 是整包的單一
    識別碼，用來快速判斷兩批產出是不是同一份程式碼跑的。
    """
    try:
        files = sorted(
            p for p in pkg_dir.rglob("*.py")
            if "__pycache__" not in p.parts
        )
        per_file, h = {}, hashlib.sha256()
        for p in files:
            data = p.read_bytes()
            rel  = p.relative_to(pkg_dir).as_posix()
            per_file[rel] = _sha256_bytes(data)[:16]
            h.update(rel.encode())
            h.update(data)
        return {"combined": h.hexdigest()[:16], "files": per_file}
    except Exception as e:                                    # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _file_fingerprint(path: Path) -> dict | None:
    """單一檔案（通常是入口腳本）的指紋。"""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return None
        return {"path": str(p), "sha256": _sha256_bytes(p.read_bytes())[:16]}
    except Exception:                                         # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────
#  環境
# ─────────────────────────────────────────────────────────────

def git_info(cwd: Path | None = None) -> dict:
    """
    git commit / 是否有未提交改動。

    ⚠ 實驗副本（見 summary_research.md §0.1）常常**不是** git repo，或是整個
    資料夾複製過來、git 資訊指向錯的地方。所以 git 只是輔助欄位，
    真正可信的識別碼是 `code_hash`（內容指紋，與 git 無關）。
    """
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=cwd, capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:                                     # noqa: BLE001
            return None

    head = _run("git", "rev-parse", "HEAD")
    if head is None:
        return {"available": False}
    status = _run("git", "status", "--porcelain")
    return {
        "available": True,
        "commit":    head[:12],
        "branch":    _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty":     bool(status),
    }


def env_info() -> dict:
    info = {
        "python":   sys.version.split()[0],
        "platform": platform.platform(),
        "cwd":      os.getcwd(),
    }
    for name in ("lightgbm", "optuna", "pandas", "numpy", "sklearn", "scipy"):
        try:
            info[name] = __import__(name).__version__
        except Exception:                                     # noqa: BLE001
            info[name] = None
    return info


def _parallel_trials_flag() -> dict:
    """
    記錄本次跑的時候 `_PARALLEL_TRIALS` 是開還是關。

    這是**可重現性的總開關**（見 summary_research.md §0.2 / §4.4）：開啟時
    TPESampler 的抽樣順序受排程影響，即使固定 seed 也不再逐次可重現。
    產出若是在開啟狀態下跑的，就不該被當成別的實驗的固定上游 —— 但光看
    predictions.csv 完全看不出來，所以必須記在 manifest 裡。
    """
    try:
        from . import lgb_utils
        return {
            "parallel_trials": bool(lgb_utils._PARALLEL_TRIALS),
            "reproducible":    not bool(lgb_utils._PARALLEL_TRIALS),
            "n_cores":         getattr(lgb_utils, "_N_CORES", None),
        }
    except Exception as e:                                    # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


# ─────────────────────────────────────────────────────────────
#  組裝 / 寫出
# ─────────────────────────────────────────────────────────────

def _jsonable(obj):
    """Path / set / numpy 純量等轉成可 JSON 序列化的形式。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):          # numpy 純量
        try:
            return obj.item()
        except Exception:             # noqa: BLE001
            pass
    return repr(obj)


def build_run_manifest(config: dict, extra: dict | None = None) -> dict:
    """組出 manifest dict。`config` 通常是 dataclasses.asdict(WalkForwardConfig)。"""
    manifest = {
        "manifest_version": 1,
        "created_at":       datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_hash":        None,
        "code":             code_fingerprint(),
        "entry_script":     _file_fingerprint(sys.argv[0]) if sys.argv and sys.argv[0] else None,
        "git":              git_info(_PKG_DIR.parent),
        "env":              env_info(),
        "reproducibility":  _parallel_trials_flag(),
        "config":           _jsonable(config),
    }
    manifest["code_hash"] = manifest["code"].get("combined")
    if extra:
        manifest.update(_jsonable(extra))
    return manifest


def write_run_manifest(
    out_dir:  Path,
    config:   dict,
    extra:    dict | None = None,
    filename: str = "run_manifest.json",
) -> Path | None:
    """寫出 manifest；任何失敗只印警告，不中斷訓練。"""
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(build_run_manifest(config, extra), f,
                      ensure_ascii=False, indent=2)
        return path
    except Exception as e:                                    # noqa: BLE001
        print(f"  ⚠ run_manifest 寫入失敗（不影響訓練結果）：{type(e).__name__}: {e}")
        return None

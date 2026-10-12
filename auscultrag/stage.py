"""Stage bookkeeping (R13): every output carries the config hash, git commit and seed that made it."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from auscultrag.config import REPO_ROOT


def git_commit() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                           capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def write_sidecar(output: Path, *, stage: str, tier: str, config: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    side = output.with_name(output.name + ".meta.json")
    payload = {
        "stage": stage,
        "tier": tier,
        "config": config,
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    side.write_text(json.dumps(payload, indent=2, default=str))
    return side


def cached(output: Path, config: dict[str, Any]) -> bool:
    """True if `output` exists and was produced by an identical config (stage can be skipped)."""
    side = output.with_name(output.name + ".meta.json")
    if not (output.exists() and side.exists()):
        return False
    return json.loads(side.read_text()).get("config_hash") == config_hash(config)

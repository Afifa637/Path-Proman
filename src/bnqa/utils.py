"""Shared plumbing: seeding, hashing, JSONL/CSV io, result logging.

Every training entry point calls :func:`set_seed`.  Every artefact written by the
project goes through :func:`write_jsonl` / :func:`log_result` so that the config
hash and seed travel with the numbers (PLAN.md §16.1, VC-12).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .config import CFG, RESULTS_JSON, config_hash

# --------------------------------------------------------------------------- #
# Console                                                                      #
# --------------------------------------------------------------------------- #


def force_utf8_stdout() -> None:
    """Windows consoles default to cp1252; printing Bangla raises UnicodeEncodeError.

    PLAN.md §4.1.  Importing bnqa calls this, so no script has to remember.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and getattr(stream, "encoding", "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - a non-reconfigurable stream is fine
            pass


# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #


def set_seed(seed: int | None = None) -> int:
    seed = CFG.seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is Tier B only — never a hard dependency of Tier A
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


# --------------------------------------------------------------------------- #
# Hashing                                                                      #
# --------------------------------------------------------------------------- #


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# JSONL io                                                                     #
# --------------------------------------------------------------------------- #


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Result logging (VC-12: no number is ever hand-entered)                       #
# --------------------------------------------------------------------------- #


def log_result(section: str, key: str, payload: dict) -> None:
    """Append one measured row to reports/results.json, stamped with hash+seed."""
    results: dict = read_json(RESULTS_JSON) if RESULTS_JSON.exists() else {}
    results.setdefault(section, {})[key] = {
        **payload,
        "config_hash": config_hash(),
        "seed": CFG.seed,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(RESULTS_JSON, results)


def stale_results(current: str | None = None) -> dict[str, list[str]]:
    """Rows in ``results.json`` that were written by a different configuration.

    VC-12 says every number in the report traces back to the configuration
    that produced it.  A row left over from an earlier config hash still
    *looks* like a current result, and is the easiest way for a stale number
    to survive into a report — so it is found by name rather than trusted to
    be noticed.
    """
    if not RESULTS_JSON.exists():
        return {}
    current = current or config_hash()
    results = read_json(RESULTS_JSON)
    out: dict[str, list[str]] = {}
    for section, rows in results.items():
        stale = [k for k, v in rows.items()
                 if isinstance(v, dict) and v.get("config_hash") != current]
        if stale:
            out[section] = stale
    return out


def prune_results(current: str | None = None) -> dict[str, list[str]]:
    """Drop stale rows.  Returns what was removed, for the caller to report."""
    stale = stale_results(current)
    if not stale:
        return {}
    results = read_json(RESULTS_JSON)
    for section, keys in stale.items():
        for k in keys:
            results[section].pop(k, None)
        if not results[section]:
            results.pop(section, None)
    write_json(RESULTS_JSON, results)
    return stale


def save_table(name: str, rows: list[dict]) -> Path:
    """Write reports/tables/<name>.csv with the config hash on every row."""
    import pandas as pd

    from .config import TABLES

    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    df = pd.DataFrame(rows)
    if len(df):
        df["config_hash"] = config_hash()
        df["seed"] = CFG.seed
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Misc                                                                         #
# --------------------------------------------------------------------------- #


@contextmanager
def timer(label: str) -> Iterator[dict]:
    box: dict = {}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - t0
        print(f"  [{label}] {box['seconds']:.2f}s")


def human_bytes(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

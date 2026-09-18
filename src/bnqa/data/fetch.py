"""Corpus acquisition + provenance ledger (PLAN.md §6.3, tasks T1/T2).

Every byte the project uses arrives through this module, and every file lands a
row in ``data/sources.csv`` carrying **URL, license, size, sha256 and download
timestamp**.  That ledger is what Sir's Q6 ("describe the collection process")
actually asks for, and it is what makes VC-11 checkable: an examiner can
re-download each file and re-hash it.

Plain HTTPS GETs.  No API key, no login, no scraping, and deliberately **no**
``datasets`` script execution — the BanglaRQA repo ships a loader script that
``datasets`` v3+ refuses to run, so we take the raw JSON instead (PLAN.md §4.1).
"""

from __future__ import annotations

import csv
import time
import urllib.request
from pathlib import Path

from ..config import SOURCES, SOURCES_CSV, TIER_A_SOURCES, ensure_dirs
from ..utils import human_bytes, sha256_file

USER_AGENT = "bnqa-research/0.4 (CSE4122 NLP lab project; +local use)"
LEDGER_FIELDS = ["name", "url", "license", "path", "bytes", "sha256",
                 "downloaded_at", "rows", "note"]


# --------------------------------------------------------------------------- #
# Download                                                                     #
# --------------------------------------------------------------------------- #


def download(url: str, dest: Path, *, expect_bytes: int | None = None,
             sha256: str | None = None, force: bool = False) -> Path:
    """Fetch ``url`` to ``dest``, resuming nothing and verifying everything.

    A pinned ``sha256`` is checked **before** any parser sees the file, so a
    corrupted or silently-changed upstream fails loudly rather than propagating
    into the corpus.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        size = dest.stat().st_size
        if expect_bytes and size != expect_bytes:
            print(f"    ! {dest.name} is {human_bytes(size)}, expected "
                  f"{human_bytes(expect_bytes)} - re-downloading")
        else:
            print(f"    = {dest.name} already present ({human_bytes(size)})")
            _verify(dest, sha256)
            return dest

    print(f"    v {dest.name} <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            got += len(chunk)
            if total:
                pct = 100 * got / total
                print(f"\r      {human_bytes(got)} / {human_bytes(total)} ({pct:4.1f}%)",
                      end="", flush=True)
    if total:
        print()
    tmp.replace(dest)
    dt = time.perf_counter() - t0
    print(f"      done in {dt:.1f}s ({human_bytes(dest.stat().st_size / max(dt, 1e-9))}/s)")

    _verify(dest, sha256)
    return dest


def _verify(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"sha256 mismatch for {path.name}\n  expected {expected}\n  actual   {actual}\n"
            "Upstream changed or the download is corrupt - refusing to parse it.")
    print(f"      sha256 OK ({expected[:16]}...)")


# --------------------------------------------------------------------------- #
# Provenance ledger                                                            #
# --------------------------------------------------------------------------- #


def _rel(path: Path) -> str:
    """Repo-relative path, so the ledger is portable between machines."""
    from ..config import ROOT

    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _read_ledger() -> dict[str, dict]:
    if not SOURCES_CSV.exists():
        return {}
    with open(SOURCES_CSV, encoding="utf-8", newline="") as fh:
        return {row["name"]: row for row in csv.DictReader(fh)}


def record_source(name: str, *, path: Path, url: str, license: str,
                  rows: int | None = None, note: str = "") -> dict:
    """Append/refresh one row of ``data/sources.csv``."""
    ledger = _read_ledger()
    ledger[name] = {
        "name": name,
        "url": url,
        "license": license,
        "path": _rel(path),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else "",
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows": "" if rows is None else rows,
        "note": note,
    }
    SOURCES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in ledger.values():
            writer.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})
    return ledger[name]


def update_rows(name: str, rows: int, note: str = "") -> None:
    """Record how many usable records a source yielded after parsing."""
    ledger = _read_ledger()
    if name not in ledger:
        return
    ledger[name]["rows"] = rows
    if note:
        ledger[name]["note"] = note
    with open(SOURCES_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in ledger.values():
            writer.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def fetch(names: list[str] | None = None, *, force: bool = False) -> None:
    ensure_dirs()
    names = names or TIER_A_SOURCES
    print(f"Fetching {len(names)} source(s)")
    for name in names:
        spec = SOURCES[name]
        print(f"  [{name}]  license={spec['license']}")
        path = download(spec["url"], spec["path"], expect_bytes=spec.get("expect_bytes"),
                        sha256=spec.get("sha256"), force=force)
        record_source(name, path=path, url=spec["url"], license=spec["license"])
    print(f"\nProvenance ledger -> {SOURCES_CSV}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Download the corpora (Tier A by default).")
    ap.add_argument("--names", nargs="*", default=None, help="source keys from config.SOURCES")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    fetch(args.names, force=args.force)


if __name__ == "__main__":
    main()

"""Evidence receipts (PLAN.md §11, VC-3 and VC-8, task V1).

Every answer exports ``reports/receipts/<qid>.json`` carrying everything needed
to re-check it **without us**: the question, the answer, the pid and character
offsets, the **sha256 of the passage text**, the evidence sentence, the
confidence, all five signals, the config hash, the seed and a timestamp.

``python -m bnqa.verify.receipt <file>`` (or ``scripts/verify_receipt.py``)
re-reads ``passages.jsonl`` on the examiner's machine, re-hashes the passage,
re-checks that the answer really is ``text[start:end]``, and prints PASS or
FAIL.  It imports nothing from the model — it is a corpus check, so it stays
true even if every model in the repo is deleted.

The passage hash is what makes this more than a formatting exercise: it pins
the receipt to an exact corpus state, so a tampered or re-built corpus cannot
quietly validate an old answer.  That is also what makes VC-7's counterfactual
test legible — under the tampered corpus the hash *should* differ, and the
receipt says so instead of failing mysteriously.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..config import CFG, PASSAGES, RECEIPTS, config_hash
from ..utils import load_jsonl, read_json, sha256_text, write_json

RECEIPT_VERSION = 1


# --------------------------------------------------------------------------- #
# Writing                                                                      #
# --------------------------------------------------------------------------- #


def build_receipt(*, qid: str, question: str, answer: str, pid: str,
                  char_start: int, char_end: int, passage_text: str,
                  evidence: str, confidence: float, signals: dict,
                  supported: bool, veto_reason: str = "",
                  citation: dict | None = None, extra: dict | None = None) -> dict:
    return {
        "receipt_version": RECEIPT_VERSION,
        "qid": qid,
        "question": question,
        "answer": answer,
        "pid": pid,
        "char_start": int(char_start),
        "char_end": int(char_end),
        "passage_sha256": sha256_text(passage_text),
        "passage_chars": len(passage_text),
        "evidence_sentence": evidence,
        "confidence": round(float(confidence), 6),
        "signals": {k: round(float(v), 6) for k, v in (signals or {}).items()},
        "supported": bool(supported),
        "veto_reason": veto_reason,
        "citation": citation or {},
        "config_hash": config_hash(),
        "seed": CFG.seed,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(extra or {}),
    }


def write_receipt(receipt: dict, *, directory: Path | None = None) -> Path:
    directory = directory or RECEIPTS
    directory.mkdir(parents=True, exist_ok=True)
    # qids contain '#' for the disambiguated duplicates (T1), which is legal in
    # a filename but awkward in a URL, so it is mapped rather than left to chance.
    safe = str(receipt["qid"]).replace("#", "_dup")
    path = directory / f"{safe}.json"
    write_json(path, receipt)
    return path


# --------------------------------------------------------------------------- #
# Checking — the part that runs on the examiner's machine                      #
# --------------------------------------------------------------------------- #


def _load_passages(passages_path: Path | None = None) -> dict[str, str]:
    path = passages_path or PASSAGES
    return {p["pid"]: p["text"] for p in load_jsonl(path)}


def check_receipt(receipt: dict, passages: dict[str, str] | None = None) -> dict:
    """Re-verify one receipt against the corpus.  Returns a per-check report."""
    passages = _load_passages() if passages is None else passages
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    pid = receipt.get("pid", "")
    text = passages.get(pid)
    add("passage_exists", text is not None, f"pid={pid}")

    if text is not None:
        actual_hash = sha256_text(text)
        add("passage_sha256", actual_hash == receipt.get("passage_sha256"),
            f"expected {str(receipt.get('passage_sha256'))[:16]}…, "
            f"got {actual_hash[:16]}…")

        start, end = receipt.get("char_start", 0), receipt.get("char_end", 0)
        in_range = 0 <= start <= end <= len(text)
        add("offsets_in_range", in_range, f"[{start}, {end}) of {len(text)} chars")

        if in_range:
            slice_ = text[start:end]
            add("span_is_substring", slice_ == receipt.get("answer"),
                f"corpus says {slice_!r}, receipt says {receipt.get('answer')!r}")
            add("evidence_contains_span",
                receipt.get("answer", "") in receipt.get("evidence_sentence", ""),
                "the answer must be inside its own evidence sentence")

    add("config_hash_matches", receipt.get("config_hash") == config_hash(),
        f"receipt {receipt.get('config_hash')} vs current {config_hash()} "
        "(a mismatch is informational, not a failure of the answer)")

    hard = [c for c in checks if c["check"] != "config_hash_matches"]
    return {"qid": receipt.get("qid"), "checks": checks,
            "passed": all(c["pass"] for c in hard),
            "n_checks": len(checks)}


def print_report(report: dict) -> bool:
    print(f"receipt {report['qid']}")
    for c in report["checks"]:
        mark = "PASS" if c["pass"] else "FAIL"
        print(f"  [{mark}] {c['check']:24s} {c['detail']}")
    verdict = "PASS" if report["passed"] else "FAIL"
    print(f"  => {verdict}")
    return report["passed"]


def check_file(path: Path) -> bool:
    return print_report(check_receipt(read_json(path)))


def check_all(directory: Path | None = None) -> dict:
    directory = directory or RECEIPTS
    passages = _load_passages()
    files = sorted(directory.glob("*.json"))
    reports = [check_receipt(read_json(f), passages) for f in files]
    passed = sum(1 for r in reports if r["passed"])
    return {"n": len(reports), "passed": passed, "failed": len(reports) - passed,
            "all_pass": passed == len(reports), "failures":
                [r["qid"] for r in reports if not r["passed"]][:20]}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-verify an evidence receipt against passages.jsonl (VC-3).")
    ap.add_argument("receipt", nargs="?", help="path to a receipt JSON; omit to check all")
    args = ap.parse_args()

    if args.receipt:
        return 0 if check_file(Path(args.receipt)) else 1

    summary = check_all()
    print(f"{summary['passed']}/{summary['n']} receipts PASS")
    if summary["failures"]:
        print("  failures:", ", ".join(summary["failures"]))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""VC-3 — re-verify an evidence receipt.  **Runs on the examiner's machine.**

    python scripts/verify_receipt.py reports/receipts/<qid>.json
    python scripts/verify_receipt.py                    # check every receipt

Re-reads ``data/processed/passages.jsonl``, re-hashes the cited passage,
re-checks that the answer really is ``text[start:end]``, and prints PASS or
FAIL.  It loads no model and needs no network — it is a corpus check, so it
stays true even if every fitted artefact in the repo is deleted.

That is the point of the receipt: the examiner does not have to trust us, or
even to be able to run us.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnqa.verify.receipt import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

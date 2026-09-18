"""Central configuration — every path and hyperparameter lives here.

PLAN.md §7.1 / Appendix B.  No constant is hard-coded in any other module: if a
script needs a number, it imports it from here.  The serialised config is hashed
(``config_hash()``) and that hash is written into every artefact and every row of
``reports/results.json``, so any number traces back to the configuration that
produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
COUNTERFACTUAL = DATA / "counterfactual"
SOURCES_CSV = DATA / "sources.csv"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
RECEIPTS = REPORTS / "receipts"
RESULTS_JSON = REPORTS / "results.json"
ENV_JSON = REPORTS / "env.json"

RESOURCES = ROOT / "src" / "bnqa" / "resources"

# raw sub-directories, one per source
RAW_BANGLARQA = RAW / "banglarqa"
RAW_NCTB = RAW / "nctb_schooltext"
RAW_TEXTBOOK = RAW / "bangla_textbook"
RAW_WIKI = RAW / "wiki"
RAW_BCORE = RAW / "bcore"  # Tier B only

# processed artefacts
PASSAGES = PROCESSED / "passages.jsonl"
QA_TRAIN = PROCESSED / "qa_train.jsonl"
QA_VAL = PROCESSED / "qa_val.jsonl"
QA_TEST = PROCESSED / "qa_test.jsonl"
INDEX_MANIFEST = {n: PROCESSED / f"index_{n // 1000}k.manifest.json" for n in (10_000, 50_000)}

ALL_DIRS = [
    DATA, RAW, PROCESSED, COUNTERFACTUAL,
    RAW_BANGLARQA, RAW_NCTB, RAW_TEXTBOOK, RAW_WIKI, RAW_BCORE,
    REPORTS, FIGURES, TABLES, RECEIPTS,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Remote sources (PLAN.md §23 — verified 17 Sep 2026, all ungated)             #
# --------------------------------------------------------------------------- #

HF = "https://huggingface.co/datasets"

SOURCES = {
    # BanglaRQA ships a loader script; datasets>=3 refuses to run it, so we take
    # the raw JSON over plain HTTPS.  Expected sizes are asserted on download.
    "banglarqa_train": dict(
        url=f"{HF}/sartajekram/BanglaRQA/resolve/main/Train.json",
        path=RAW_BANGLARQA / "Train.json", license="CC BY-NC-SA 4.0", expect_bytes=27_818_012),
    "banglarqa_val": dict(
        url=f"{HF}/sartajekram/BanglaRQA/resolve/main/Validation.json",
        path=RAW_BANGLARQA / "Validation.json", license="CC BY-NC-SA 4.0", expect_bytes=3_493_591),
    "banglarqa_test": dict(
        url=f"{HF}/sartajekram/BanglaRQA/resolve/main/Test.json",
        path=RAW_BANGLARQA / "Test.json", license="CC BY-NC-SA 4.0", expect_bytes=3_603_506),
    # sha256 pinned from the Mendeley public API on 17 Sep 2026 — a mismatch is
    # caught before we parse a single byte.
    "nctb_schooltext": dict(
        url="https://data.mendeley.com/public-files/datasets/f3882ccczp/files/"
            "54e66048-5d2b-47a0-a489-a0b104119476/file_downloaded",
        path=RAW_NCTB / "NCTB-SchoolText.zip", license="CC BY 4.0", expect_bytes=12_202_106,
        sha256="10ce1f2d8b40b6cf600a067b9916e042a386c0e52d881dc518442b6376665b31"),
    "bangla_textbook": dict(
        url=f"{HF}/md-nishat-008/Bangla-TextBook/resolve/main/bangla_textbook_128w_cleaned.csv",
        path=RAW_TEXTBOOK / "bangla_textbook_128w_cleaned.csv", license="MIT",
        expect_bytes=188_650_371),
    "wiki_bn_00": dict(
        url=f"{HF}/wikimedia/wikipedia/resolve/main/20231101.bn/train-00000-of-00002.parquet",
        path=RAW_WIKI / "train-00000-of-00002.parquet", license="CC BY-SA 4.0"),
    "wiki_bn_01": dict(
        url=f"{HF}/wikimedia/wikipedia/resolve/main/20231101.bn/train-00001-of-00002.parquet",
        path=RAW_WIKI / "train-00001-of-00002.parquet", license="CC BY-SA 4.0"),
}

# Tier A needs only these five.  wiki_bn_01 and B-CORE are Tier B / optional.
TIER_A_SOURCES = ["banglarqa_train", "banglarqa_val", "banglarqa_test",
                  "nctb_schooltext", "bangla_textbook", "wiki_bn_00"]


# --------------------------------------------------------------------------- #
# Hyperparameters                                                              #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Config:
    seed: int = 1337

    # ---- corpus -----------------------------------------------------------
    min_passage_chars: int = 40
    min_bengali_ratio: float = 0.60
    max_passage_chars: int = 2_000
    dedup_jaccard: float = 0.85
    dedup_num_perm: int = 64
    index_sizes: tuple[int, ...] = (10_000, 50_000)
    # nested composition: gold is present in full at every size
    index_composition: dict = field(default_factory=lambda: {
        10_000: {"gold": 3_000, "wiki": 3_800, "nctb": 3_200},
        50_000: {"gold": 3_000, "wiki": 27_000, "nctb": 20_000},
    })
    wiki_chunk_tokens: tuple[int, int] = (120, 180)
    wiki_chunk_overlap: int = 30
    nctb_grades: tuple[int, ...] = (6, 7, 8, 9, 10)

    # ---- sparse retrieval (T5/T7) ----------------------------------------
    bm25_k1: float = 1.2          # tuned on val in T7
    bm25_b: float = 0.75
    tfidf_word_ngrams: tuple[int, int] = (1, 2)
    tfidf_char_ngrams: tuple[int, int] = (3, 5)
    tfidf_min_df: int = 2
    tfidf_max_features_word: int = 400_000
    tfidf_max_features_char: int = 600_000
    rrf_k: int = 60
    top_k: int = 10
    query_expansion_n: int = 2

    # ---- reader (T10) -----------------------------------------------------
    cand_sentences: int = 3
    cand_ngram_max: int = 12
    span_max_tokens: int = 30

    # ---- verification (T11) ----------------------------------------------
    conformal_alpha: float = 0.10
    conformal_delta: float = 0.10
    contrast_set_size: int = 500
    banglaverify_verify_sample: int = 300

    # ---- evaluation -------------------------------------------------------
    bootstrap_resamples: int = 1_000
    headline_index: int = 50_000
    headline_metric: str = "token_f1"
    recall_at: tuple[int, ...] = (1, 5, 10)
    counterfactual_n: int = 30
    grading_study_n: int = 60
    graders: int = 3

    # ---- T7 topic figure --------------------------------------------------
    kmeans_clusters: tuple[int, ...] = (5, 8, 12, 16, 20)
    kmeans_sample: int = 8_000

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


CFG = Config()


def config_hash(cfg: Config = CFG) -> str:
    """Stable 12-hex-char digest of the whole configuration."""
    blob = json.dumps(cfg.as_dict(), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

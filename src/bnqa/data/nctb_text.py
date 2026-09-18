"""NCTB textbook ingest — no PDFs, no OCR (PLAN.md task T2).

The proposal planned PDF download → PyMuPDF → Tesseract fallback → chunking.
All of it is replaced by two already-published clean-text corpora, which removes
the riskiest hours of the sprint and an entire error class: Bangla PDF
extraction mangles conjunct glyphs through font-encoding bugs, and that noise
would have propagated into every reported number (PLAN.md §4, deviation 3).

**NCTB-SchoolText** (Mendeley, CC BY 4.0) is the primary source and the one that
matters, because each chunk carries ``class / subject / chapter_no /
chapter_title``.  That metadata *is* our citation unit — the
``শ্রেণি ৮ › বিজ্ঞান › অধ্যায় ৪`` line under every answer (VC-2) — so we keep
**their** chunking rather than re-chunking, and credit their rule as theirs.

**Bangla-TextBook** (HuggingFace, MIT) is the documented fallback.  Verified in
T2: it ships a single ``text`` column with **no chapter metadata**, so it can
supply passages but not citations.  It is ingested and logged for provenance,
and used only to top up the index if SchoolText runs short.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from ..config import (CFG, PROCESSED, RAW_NCTB, RAW_TEXTBOOK, SOURCES, ensure_dirs)
from ..preprocess.normalize import bengali_ratio, normalize
from ..utils import log_result, save_table, write_jsonl

CLASS_DIR_RE = re.compile(r"^class([A-Za-z]+)/processed_chapters_([^/]+)/.*\.jsonl$")

#: folder stem -> grade.  ``classNineTen`` covers grades 9 and 10 in one book,
#: so its records keep whatever ``class`` value the record itself carries.
CLASS_TO_GRADE = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
    "Six": 6, "Seven": 7, "Eight": 8, "NineTen": 9,
}

SUBJECT_BN = {
    "Bangla": "বাংলা", "English": "ইংরেজি", "Math": "গণিত", "Mathematics": "গণিত",
    "Science": "বিজ্ঞান", "BGS": "বাংলাদেশ ও বিশ্বপরিচয়", "ICT": "তথ্য ও যোগাযোগ প্রযুক্তি",
    "Religion": "ধর্ম", "Agriculture": "কৃষিশিক্ষা", "HomeScience": "গার্হস্থ্যবিজ্ঞান",
    "PhysicalEducation": "শারীরিক শিক্ষা", "Arts": "চারু ও কারুকলা",
    "Physics": "পদার্থবিজ্ঞান", "Chemistry": "রসায়ন", "Biology": "জীববিজ্ঞান",
    "HigherMath": "উচ্চতর গণিত", "Economics": "অর্থনীতি", "Geography": "ভূগোল",
    "History": "ইতিহাস", "CivicScience": "পৌরনীতি", "Accounting": "হিসাববিজ্ঞান",
    "BusinessEntrepreneurship": "ব্যবসায় উদ্যোগ", "Finance": "ফিন্যান্স",
}


def _keep(text: str) -> bool:
    """Corpus filter of PLAN.md §6.3 step 4."""
    return (len(text) >= CFG.min_passage_chars
            and bengali_ratio(text) >= CFG.min_bengali_ratio)


def parse_grade(value: object) -> tuple[int, int, str] | None:
    """``class`` -> ``(low, high, label)``.

    NCTB publishes one combined book for classes 9 and 10, and the archive
    encodes that as the **string** ``"9-10"`` rather than an integer — 27,059
    records, the single largest slice of the corpus and the one most relevant to
    a secondary-school QA demo.  A naive ``int()`` cast drops all of them
    silently, which is exactly the kind of loss an audit is supposed to catch.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value, value, str(value)
    text = str(value).strip()
    if not text:
        return None
    m = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", text)
    if m:
        low, high = int(m.group(1)), int(m.group(2))
        return min(low, high), max(low, high), f"{min(low, high)}-{max(low, high)}"
    if text.isdigit():
        return int(text), int(text), text
    return None


# --------------------------------------------------------------------------- #
# NCTB-SchoolText                                                              #
# --------------------------------------------------------------------------- #


def read_schooltext(zip_path: Path | None = None, *, grades: tuple[int, ...] | None = None
                    ) -> tuple[list[dict], Counter]:
    """Yield normalised passages from the Mendeley archive."""
    zip_path = zip_path or RAW_NCTB / "NCTB-SchoolText.zip"
    grades = grades or CFG.nctb_grades
    stats: Counter = Counter()
    rows: list[dict] = []
    seen_pids: set[str] = set()

    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.endswith(".jsonl")]
        stats["chapter_files"] = len(members)
        for name in members:
            m = CLASS_DIR_RE.match(name)
            folder_grade = CLASS_TO_GRADE.get(m.group(1)) if m else None
            with zf.open(name) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        stats["bad_json"] += 1
                        continue
                    stats["records_read"] += 1

                    parsed = parse_grade(rec.get("class")) or (
                        (folder_grade, folder_grade, str(folder_grade)) if folder_grade else None)
                    if parsed is None:
                        stats["no_grade"] += 1
                        continue
                    low, high, grade_label = parsed
                    if not any(g in grades for g in range(low, high + 1)):
                        stats["grade_filtered"] += 1
                        continue
                    grade = low

                    text = normalize(rec.get("text") or "")
                    if not _keep(text):
                        stats["filtered_short_or_nonbengali"] += 1
                        continue
                    if len(text) > CFG.max_passage_chars:
                        text = text[: CFG.max_passage_chars]
                        stats["truncated"] += 1

                    chunk_id = str(rec.get("chunk_id") or f"{name}:{stats['records_read']}")
                    pid = "nctb_" + re.sub(r"[^A-Za-z0-9]+", "_", chunk_id).strip("_")
                    if pid in seen_pids:
                        stats["duplicate_chunk_id"] += 1
                        pid = f"{pid}_{stats['duplicate_chunk_id']}"
                    seen_pids.add(pid)

                    subject = str(rec.get("subject") or "").strip()
                    rows.append({
                        "pid": pid,
                        "text": text,
                        "source": "nctb_schooltext",
                        "grade": grade,
                        "grade_label": grade_label,
                        "subject": subject,
                        "subject_bn": SUBJECT_BN.get(subject, subject),
                        "chapter_no": rec.get("chapter_no"),
                        "chapter_title": normalize(str(rec.get("chapter_title") or "")),
                        "title": None,
                    })
                    stats["kept"] += 1
                    stats[f"grade_{grade_label}"] += 1
    return rows, stats


# --------------------------------------------------------------------------- #
# Bangla-TextBook (fallback / provenance)                                      #
# --------------------------------------------------------------------------- #


def read_bangla_textbook(csv_path: Path | None = None, *, limit: int | None = None
                         ) -> tuple[list[dict], Counter]:
    """Read the 128-word-chunked CSV.  No chapter metadata is available."""
    import pandas as pd

    csv_path = csv_path or RAW_TEXTBOOK / "bangla_textbook_128w_cleaned.csv"
    stats: Counter = Counter()
    rows: list[dict] = []
    if not csv_path.exists():
        stats["missing"] = 1
        return rows, stats

    for chunk in pd.read_csv(csv_path, chunksize=20_000, usecols=["text"]):
        for raw in chunk["text"].astype(str):
            stats["records_read"] += 1
            text = normalize(raw)
            if not _keep(text):
                stats["filtered_short_or_nonbengali"] += 1
                continue
            if len(text) > CFG.max_passage_chars:
                text = text[: CFG.max_passage_chars]
                stats["truncated"] += 1
            rows.append({
                "pid": f"btb_{stats['kept']:06d}",
                "text": text,
                "source": "bangla_textbook",
                "grade": None, "grade_label": None, "subject": None, "subject_bn": None,
                "chapter_no": None, "chapter_title": None, "title": None,
            })
            stats["kept"] += 1
            if limit and stats["kept"] >= limit:
                return rows, stats
    return rows, stats


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def build(*, textbook_limit: int = 40_000) -> dict:
    ensure_dirs()
    print("T2  NCTB textbook ingest (no PDFs, no OCR)")

    school, s_stats = read_schooltext()
    print(f"  NCTB-SchoolText: {s_stats['chapter_files']:,d} chapter files -> "
          f"{s_stats['records_read']:,d} chunks read, {len(school):,d} kept "
          f"(grades {min(CFG.nctb_grades)}-{max(CFG.nctb_grades)})")
    print(f"    dropped: {s_stats['grade_filtered']:,d} out-of-grade, "
          f"{s_stats['filtered_short_or_nonbengali']:,d} short/non-Bengali")
    by_grade = {k.replace("grade_", ""): v for k, v in sorted(s_stats.items())
                if k.startswith("grade_") and k != "grade_filtered" and v}
    print(f"    by grade: {by_grade}")
    subjects = Counter(r["subject"] for r in school)
    print(f"    {len(subjects)} subjects, top: {dict(subjects.most_common(6))}")

    tb, t_stats = read_bangla_textbook(limit=textbook_limit)
    if t_stats.get("missing"):
        print("  Bangla-TextBook: not downloaded (optional fallback) - skipped")
    else:
        print(f"  Bangla-TextBook: {len(tb):,d} kept of {t_stats['records_read']:,d} read "
              f"(capped at {textbook_limit:,d}); NO chapter metadata, so it can supply "
              f"passages but not citations")

    rows = school + tb
    out = PROCESSED / "nctb_passages.jsonl"
    write_jsonl(out, rows)
    print(f"  -> {len(rows):,d} passages to {out.relative_to(out.parents[2])}")

    save_table("nctb_ingest", [
        {"source": "nctb_schooltext", "chapter_files": s_stats["chapter_files"],
         "records_read": s_stats["records_read"], "kept": len(school),
         "grade_filtered": s_stats["grade_filtered"],
         "filtered_short_or_nonbengali": s_stats["filtered_short_or_nonbengali"],
         "subjects": len(subjects), "has_citation_metadata": True},
        {"source": "bangla_textbook", "chapter_files": 0,
         "records_read": t_stats["records_read"], "kept": len(tb),
         "grade_filtered": 0,
         "filtered_short_or_nonbengali": t_stats["filtered_short_or_nonbengali"],
         "subjects": 0, "has_citation_metadata": False},
    ])

    payload = {
        "schooltext_kept": len(school),
        "schooltext_chapters": s_stats["chapter_files"],
        "schooltext_subjects": len(subjects),
        "schooltext_by_grade": by_grade,
        "bangla_textbook_kept": len(tb),
        "total": len(rows),
    }
    log_result("t2_nctb", "ingest", payload)

    from .fetch import update_rows

    update_rows("nctb_schooltext", len(school),
                note=f"{s_stats['chapter_files']} chapters, grade groups "
                     f"{','.join(by_grade) if by_grade else '-'}")
    if not t_stats.get("missing"):
        update_rows("bangla_textbook", len(tb), note="no chapter metadata; capped")
    return payload


if __name__ == "__main__":
    build()

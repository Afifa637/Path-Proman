# TASKS — Tier A build log

Status of the tasks in `PLAN.md` §14. One task at a time; **done** means the
task's *Done when* criterion in PLAN.md §14 is met and the artefact exists.

| Task | Status | Artefacts | Notes |
|---|---|---|---|
| **T0** Repo skeleton, config, seeds, env | ✅ done | `src/bnqa/`, `config.py`, `reports/env.json`, `requirements-core.txt` | Tier A installs with **no torch / transformers / gensim** |
| **T1** BanglaRQA ingest, leakage gate, alignment audit | ✅ done | `qa_{train,val,test}.jsonl`, `gold_contexts.jsonl`, `tables/splits.csv`, `tables/answer_alignment.csv` | counts match 11,912 / 1,484 / 1,493; leakage gate **CLEAN**; span alignment **97.9%** |
| **T2** NCTB textbook ingest | ✅ done | `nctb_passages.jsonl`, `tables/nctb_ingest.csv`, `data/sources.csv` | 43,359 SchoolText passages, grades 6–10, 32 subjects |
| **T3** Index assembly + dedup + audit | ✅ done | `passages.jsonl`, `index_{10,50}k.manifest.json`, `tables/corpus_audit.csv` | nested, all 3,000 gold at both sizes, 20,000 citable NCTB passages |
| **T4** Normalisation, tokeniser, stopwords, stemmer | ✅ done | `bnqa/preprocess/` | idempotent; stemmer is corpus-validated |
| **T5** Sparse retrieval (TF-IDF word/char, our BM25) | ✅ done | `bnqa/retrieval/`, `tables/retrieval.csv` | BM25 validated against a hand-worked example |
| **T6b** Lexical resources + probe | ✅ done | `resources/{suffixes,synonyms,variants,synonym_probe}`, `tables/synonym_probe.csv` | precision **1.000**, recall **0.714** on held-out pairs |
| **T7** RRF hybrid, query expansion, index-size comparison, topics | ✅ done | `tables/retrieval.csv`, `tables/retrieval_index_size.csv`, `figures/topics_50k.png` | RQ1 answered on val at both sizes |
| T10 Reader | ⬜ next | | |
| T11 Verifier + T11b BanglaVerify | ⬜ | | |
| T12 Pipeline + 3-tab Streamlit | ⬜ | | |
| V1–V4 Verifiability | ⬜ | | |
| ~~G1–G3 Grading~~ | ❌ removed | | out of scope — PLAN.md §12 |

## Findings that changed the code

Every one of these was found by running the thing, not by planning it.

1. **BanglaRQA has no `answer_start`, and the naive alignment rate looks like a
   failure.** Pooling all answers gives 82.7%, below PLAN.md's 85% floor. Two
   causes, both real and both worth reporting:
   - `yes/no` answers (2,502) are *labels*, not spans — they cannot align by
     construction and do not belong in a span-alignment denominator;
   - `multiple spans` answers are **semicolon-delimited lists**; the joined
     string occurs nowhere in the passage but each part occurs verbatim.
     Splitting on `;` took multi-span alignment from **53.1% → 98.6%**.

   Measured over span-typed answers the rate is **97.9% (96.2% strict)**, so the
   span-level reader is viable and PLAN.md §17's sentence-level fallback is not
   needed.

2. **Span text must be the passage's characters, not the gold answer string.**
   They differ whenever alignment was whitespace-tolerant or fuzzy, which made
   VC-1's substring assertion false for exactly those spans.

3. **`class` is the string `"9-10"`** for the combined Nine–Ten book — 27,059
   records, the largest and most demo-relevant slice of NCTB-SchoolText. A naive
   `int()` cast dropped every one of them silently.

4. **Longest-suffix stemming splits inflected forms instead of converging them.**
   `উদ্ভিদের` ends with the plural genitive `দের`, so longest-match yields
   `উদ্ভি` ≠ `stem("উদ্ভিদ")` — worse than not stemming at all. The stemmer is now
   **corpus-validated**: it accepts the longest strip whose result is an attested
   token in our own corpus. No external lexicon, so Tier A stays self-contained.

5. **Dedup must protect gold passages from *both* duplicate checks.** BanglaRQA
   contains contexts that are byte-identical under different `passage_id`s;
   dropping either leaves questions pointing at a pid that is not in the index.

6. **Char 5-gram MinHash does not finish on this corpus.** Word 3-shingles give
   the same near-duplicate signal in 2.5 minutes instead of hours.

7. **7 upstream `question_id`s are reused for different questions** (within-split
   only — no cross-split leakage). Disambiguated with a `#n` suffix; `source_qid`
   keeps the original.

8. **The index had a length artefact, and it was ours.** Chunking Wikipedia to a
   fixed 120–180 tokens while leaving BanglaRQA contexts whole made the two
   trivially separable: **passage length alone predicted "is gold" at AUC 0.795**.
   That is exactly the distribution artefact PLAN.md §6.2B warns about, and it is
   not cosmetic — BM25's `b` tunes length normalisation directly, and on the
   broken index it tuned to `b=0.3` (weak normalisation, which favours long
   documents, which were the gold ones). Sampling each distractor's target length
   from the gold distribution and dropping short tails took it to **AUC 0.545**,
   essentially chance. The index was rebuilt and every retrieval number re-run.

9. **The hand-authored lexicon covers 1.54% of question tokens**, so query
   expansion fires on 8.2% of queries and changes nothing measurable. This is the
   honest Tier-A baseline that Tier B's induced synonyms (X3) have to beat — it
   makes RQ-A measurable rather than rhetorical.

10. **The synonym probe was circular** until held-out pairs were added. Every
    positive had been copied from the lexicon, so recall was 1.0 by construction.
    With 12 genuine Bangla doublets the lexicon has never seen, the honest
    numbers are **precision 1.000, recall 0.714**.

## Commands

```powershell
. .\scripts\env.ps1                        # UTF-8 console + PYTHONPATH + HF_HOME

python -m bnqa.envinfo                     # T0  -> reports/env.json
python -m bnqa.data.fetch                  # download the Tier A corpora (~550 MB)
python -m bnqa.data.banglarqa              # T1
python -m bnqa.data.nctb_text              # T2
python -m bnqa.data.wiki                   # T3a length-matched distractors
python -m bnqa.data.index_build            # T3  nested indexes
python -m bnqa.data.audit                  # T3  corpus audit
python -m bnqa.eval.lexicon_probe          # T6b
python scripts/run_retrieval.py            # T5 + T7
python -m bnqa.eval.topics                 # T7  topic figure
pytest                                     # all invariants
```

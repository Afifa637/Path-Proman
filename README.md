# পাঠ-প্রমাণ (Path-Proman)

**Evidence-grounded Bangla textbook QA with verifiable corpus attribution.**
CSE 4122 — NLP Lab project. See [`PLAN.md`](PLAN.md) for the full design and
[`TASKS.md`](TASKS.md) for build status.

> উত্তর নয় — প্রমাণসহ উত্তর. *Not an answer. An answer with proof.*

Every answer this system gives is a **literal character span of a named corpus
passage**, cited down to grade → subject → chapter → passage id → character
offsets, with a calibrated confidence and an explicit refusal when the corpus
does not support an answer.

## Two tiers

| | Tier A — the guaranteed system | Tier B — the research extension |
|---|---|---|
| Corpus, index, sparse retrieval, reader, **verifier + veto layer + conformal abstention**, counterfactual attribution, UI | **CPU only.** No torch, no transformers, no gensim. | ~500M-token pretraining corpus, our SentencePiece tokenizer, Word2Vec/FastText, a 6-layer ~25M-parameter transformer, and the neural retrieval/reader arms. Needs CUDA. |

**Tier A is complete.** Every Tier-B component replaces exactly one Tier-A
component behind a shared interface (`Retriever`, `Reader`, the verification
signal signature), so **Tier A is Tier B's ablation baseline** and the demo can
never be broken by a training run.

## Quick start (Tier A — runs on any machine)

```powershell
. .\scripts\env.ps1                  # UTF-8 console, PYTHONPATH, HF_HOME
pip install -r requirements-core.txt

.\scripts\run_all.ps1                # everything: corpus -> models -> reports
.\scripts\run_all.ps1 -Quick         # ten-minute pass that proves the wiring

streamlit run app/streamlit_app.py   # the three-tab demo
```

Or one task at a time:

```powershell
python -m bnqa.envinfo               # T0   records the machine into reports/env.json
python -m bnqa.data.fetch            # T0   ~550 MB, five public licensed corpora
python -m bnqa.data.banglarqa        # T1   supervision + leakage gate + alignment audit
python -m bnqa.data.nctb_text        # T2   NCTB textbook passages
python -m bnqa.data.wiki             # T3a  length-matched distractors
python -m bnqa.data.index_build      # T3   nested 10k / 50k indexes
python -m bnqa.data.audit            # T3   corpus audit
python -m bnqa.eval.lexicon_probe    # T6b  lexicon precision on a held-out probe
python scripts/run_retrieval.py      # T5+T7  the RQ1 comparison, both index sizes
python -m bnqa.eval.topics           # T7   K-Means + silhouette figure
python scripts/run_reader.py         # T10  span reader + feature-importance table
python scripts/run_verify.py         # T11  BanglaVerify, fusion, calibration, conformal
python scripts/run_counterfactual.py # V3   CAR / AoRR (VC-7)
python scripts/make_report_assets.py # T13  receipts, audit sheet, MODEL_CARD.md
pytest                               # every invariant
```

## The Verifiability Contract

Fourteen claims an examiner can check without trusting us. The ones that are
code rather than prose:

| | Claim | Check it yourself |
|---|---|---|
| **VC-1** | The answer is *always* a literal span of a corpus passage | `pytest tests/test_span_is_substring.py`; the UI prints the assertion |
| **VC-3** | Offline-recheckable evidence receipt | `python scripts/verify_receipt.py reports/receipts/<qid>.json` |
| **VC-4** | Corpus explorer — ask against a passage *you* picked | Tab 2 |
| **VC-5** | Abstains when the corpus has no answer | ask something outside the corpus |
| **VC-6** | Air-gapped run | `$env:BNQA_OFFLINE=1`, then unplug the Wi-Fi |
| **VC-7** | 🏆 Counterfactual corpus test — the answer follows the **book** | sidebar toggle **কর্পাস: আসল / পরিবর্তিত** |
| **VC-8** | Determinism | ask twice, compare |
| **VC-9** | Teacher audit sheet | `reports/tables/audit_sample.csv` |
| **VC-10** | Model provenance | [`MODEL_CARD.md`](MODEL_CARD.md) |
| **VC-11** | Reproducible from clean | `.\scripts\run_all.ps1` |
| **VC-12** | No hand-entered numbers | every row in `reports/` carries a config hash and seed |
| **VC-14** | A stated error guarantee | `reports/tables/conformal.csv` |

## Corpora

All public, licensed, ungated and hash-pinned in `data/sources.csv`
(PLAN.md §23). Nothing is scraped and nothing is hand-made.

| Corpus | License | Role |
|---|---|---|
| BanglaRQA | CC BY-NC-SA 4.0 | the only labelled supervision |
| NCTB-SchoolText | CC BY 4.0 | textbook passages **with grade/subject/chapter** — the citation unit |
| Bangla-TextBook | MIT | second textbook source (no chapter metadata) |
| Bangla Wikipedia | CC BY-SA 4.0 | same-distribution distractors |
| B-CORE | CC BY 4.0 | Tier B pretraining corpus |

`data/processed/banglaverify_*.jsonl` is a derivative of BanglaRQA and
therefore inherits **CC BY-NC-SA 4.0**.

## Layout

```
src/bnqa/
  config.py        every path and hyperparameter, one place
  pipeline.py      BanglaQA — the facade the CLI, app and notebooks use
  offline.py       the BNQA_OFFLINE socket guard (VC-6)
  data/            fetch · banglarqa · nctb_text · wiki · dedup · index_build · audit
  preprocess/      normalize · tokenize · stopwords · stem
  retrieval/       base (the Retriever protocol) · tfidf · bm25 · hybrid · index
  reader/          base (the Reader protocol) · qtype · candidates · features · span_ranker
  verify/          constraints (the veto layer) · signals · banglaverify · support ·
                   fusion · calibration · conformal · threshold · receipt
  eval/            retrieval_metrics · qa_metrics · verify_metrics · counterfactual ·
                   lexicon_probe · topics
  resources/       suffixes.txt · synonyms.tsv · variants.tsv · synonym_probe.tsv
app/               streamlit_app.py — the three-tab demo
scripts/           env.ps1 · run_all.ps1 · run_retrieval · run_reader · run_verify ·
                   run_counterfactual · make_report_assets · verify_receipt
reports/           tables/ · figures/ · receipts/ · results.json · env.json
tests/             corpus invariants · BM25 vs hand-worked values · veto isolation ·
                   VC-1 substring · determinism and the offline guard
```

`reports/results.json` is written **only** by evaluation scripts, and every row
carries the config hash and seed that produced it — no number in the report is
ever typed by hand.

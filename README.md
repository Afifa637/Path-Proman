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
| Corpus, index, sparse retrieval, reader, **verifier + veto layer + conformal abstention**, counterfactual attribution, grading, UI | **CPU only.** No torch, no transformers, no gensim. | ~500M-token pretraining corpus, our SentencePiece tokenizer, Word2Vec/FastText, a 6-layer ~25M-parameter transformer, and the neural retrieval/reader arms. Needs CUDA. |

Every Tier-B component replaces exactly one Tier-A component behind a shared
interface (`Retriever`, `Reader`, the verification signal signature), so **Tier A
is Tier B's ablation baseline** and the demo can never be broken by a training
run.

## Quick start (Tier A — runs on any machine)

```powershell
. .\scripts\env.ps1                  # UTF-8 console, PYTHONPATH, HF_HOME
pip install -r requirements-core.txt

python -m bnqa.envinfo               # records the machine into reports/env.json
python -m bnqa.data.fetch            # ~550 MB, five public licensed corpora
python -m bnqa.data.banglarqa        # T1  supervision + leakage gate
python -m bnqa.data.nctb_text        # T2  NCTB textbook passages
python -m bnqa.data.wiki             # T3a same-distribution distractors
python -m bnqa.data.index_build      # T3  nested 10k / 50k indexes
python -m bnqa.data.audit            # T3  corpus audit
python scripts/run_retrieval.py      # T5 + T7  the RQ1 comparison
pytest                               # corpus invariants + BM25 vs hand-worked values
```

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

## Layout

```
src/bnqa/
  config.py        every path and hyperparameter, one place
  data/            fetch · banglarqa · nctb_text · wiki · dedup · index_build · audit
  preprocess/      normalize · tokenize · stopwords · stem
  retrieval/       base (the Retriever protocol) · tfidf · bm25 · hybrid · index
  resources/       suffixes.txt · synonyms.tsv · variants.tsv · synonym_probe.tsv
  eval/            retrieval_metrics · lexicon_probe · topics
reports/           tables/ · figures/ · results.json · env.json
```

`reports/results.json` is written **only** by evaluation scripts, and every row
carries the config hash and seed that produced it — no number in the report is
ever typed by hand.

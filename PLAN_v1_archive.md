# Bangla Educational QA — Build Plan

A closed-domain Bangla question-answering system over NCTB textbook content: retrieve the evidence, answer
only from it, verify that the answer is actually supported, and abstain when it is not. Every output is a
triple — **answer, supporting sentence, calibrated confidence**.

> We work **one task at a time** — you say "do T3", I do T3 and stop.

---

## 1. The one sentence this project has to defend

Every project needs a result you can state in one sentence and hold under questioning. Ours:

> **Adding evidence-based verification with abstention cuts unsupported answers from X% to Y% at Z% coverage —
> and the same mechanism rejects contradictory answers that similarity-based verification accepts.**

This is RQ2 and the proposal's stated distinguishing contribution. Critically, **it holds regardless of how
well our from-scratch models perform in absolute terms**, because it measures the *delta* from adding
verification, not the absolute quality of the retriever or reader. In a one-week project that is the
difference between a defensible result and a gamble.

Two supporting results, both safe to report even if they come out negative:

- **RQ1** — lexical vs static-dense vs contextual retrieval, compared on one index under identical conditions.
  A finding either way is a finding.
- **From-scratch vs pretrained** — our ~25M-parameter encoder trained on 500M tokens against BanglaBERT's 2.5B.
  Expected to lose; the measured gap, reported with tokens-seen and parameter counts, *is* the result.

---

## 2. Context

Two proposal PDFs in this folder describe a CSE 4122 (NLP Lab) project. Nothing is built yet. Submission is
**next week**, and the requirement is that the **models are built from scratch** (corpora may be downloaded).

Four findings from planning research materially change the proposal:

1. **NCTB-QA (arXiv:2603.05462), the proposal's primary dataset, cannot be obtained.** The paper states:
   *"The dataset used in this study is not publicly available as it is currently being utilized in ongoing
   research projects."* No GitHub / HuggingFace / Zenodo release exists.
2. **BanglaRQA is fully available** (`sartajekram/BanglaRQA`, CC-BY-NC-SA-4.0): 3,000 contexts, 14,889 QA
   pairs, split 11,912 / 1,484 / 1,493, with an `is_answerable` flag, four `question_type`s (confirmation,
   factoid, causal, list) and three `answer_type`s (yes/no, single span, multiple spans). The
   answerable/unanswerable mix is what makes abstention trainable and measurable.
   **Its contexts are Bangla Wikipedia, not NCTB textbooks** — a real deviation, handled in §6.
3. **NCTB textbook content is already published as clean text — no PDF work is needed.** The proposal's §7.1
   plans PDF download → PyMuPDF → Tesseract OCR fallback → chunking → audit. All of it is replaced by:
   - **NCTB-SchoolText** (Mendeley, CC BY 4.0) — 58,872 passage chunks, 1,535 chapters, 34 subjects, classes
     1–10, JSONL, each record carrying grade, subject, chapter number, chapter title, a deterministic chunk id
     and cleaned text. Median chunk 351 characters. This *is* our retrieval unit and our citation, pre-built.
   - **`md-nishat-008/Bangla-TextBook`** (HuggingFace, MIT) — 163 NCTB textbooks, grades 6–12, 87,110 rows.

   This removes the riskiest hours in the sprint and an entire error class: Bangla PDF extraction mangles
   conjunct glyphs through font-encoding bugs, and that noise would have propagated into every reported number.
4. **Openly licensed Bangla corpora exist at billion-token scale**, so the from-scratch models are limited by
   our GPU budget, not by available text (§4).

### Decisions taken (user-confirmed)

| Question | Decision |
|---|---|
| What "from scratch" means | Train **our own** tokenizer, Word2Vec, FastText, transformer encoder and span reader from random init. Pretrained BanglaBERT/LaBSE appears **only as a clearly-labelled reference arm**. |
| Data | BanglaRQA supervised QA + NCTB-SchoolText passages + Bangla Wikipedia distractors + a large general Bangla corpus for training representations. |
| Deliverable | Git repo with an importable `src/bnqa` package; notebooks and the Streamlit demo both import it (one code path). |
| Scope | Core = QA pipeline with verification. Student-answer grading is an **extension** (E2), not a core claim — see §3. |
| Timeline | ~1 week, driven task-by-task by you. |

### Verified environment

RTX 3080 Laptop, 8.6 GB VRAM · Python 3.13.13 · torch 2.11.0+cu128 (CUDA available) · transformers 5.5.3 ·
gensim 4.4.0 · sklearn 1.8.0 · pandas, numpy, datasets, streamlit — all present.
**To install (T0):** `sentencepiece`, `faiss-cpu`, `bert-score`, `matplotlib`, `seaborn`, `tqdm`, `datasketch`.
Deliberately **not** installing `rank_bm25` — BM25 is written by us (T5), which satisfies "from scratch" and
removes a dependency. PyMuPDF/pdfplumber are installed but no longer on the critical path.

---

## 3. Scope: core vs extension

**Core — must be finished and evaluated:** T0–T14.
**Extensions — built only if the core is complete:** E1, E2.

Four trims fit the work into one week. Each keeps the argument and drops machinery:

| Trim | Kept | Dropped |
|---|---|---|
| Index scaling | Two sizes, 10k and 50k — still empirical | Four sizes; halves the evaluation runs |
| Metric ladder | Four tiers (§5) | Three redundant middle tiers |
| Hybrid retrieval | Reciprocal rank fusion | Weighted score interpolation |
| Student-answer grading | **Its best idea, moved into the verifier** | The separate grading deliverable |

### Why grading is an extension, and what survives

Demoted because it was not in the proposal your instructors approved; because its labels would be synthetic —
generated perturbations, not teacher judgements — which is exactly where an examiner will press; and because it
costs ~7 hours plus integration in a week that is already full.

But the contradiction insight was never really a *grading* feature. The verifier already asks *"is this answer
supported by this evidence?"*, which is the same discrimination as *"is this student's answer right?"*. So these
move into **T11, the core verifier**, where they belong:

- the **veto layer** — numerals, dates, units, entities, polarity, relation order;
- the **contrast set** of minimal pairs, reported separately;
- **minimally-contrasting negatives** for the S3 support classifier;
- the **fuzzy-matching isolation rule** and its test.

They produce the second half of the headline sentence. E2 then becomes a thin UI over a verifier that already
does the hard part — which is why it is cheap to add back if the core finishes early.

---

## 4. Corpus strategy — "rich" means three different things

Corpus size acts in a **different direction** in each of three corpora. Conflating them wastes a week.

### A. Training corpus for the from-scratch representations — richer is better, to a computable limit

Feeds the tokenizer (T4), Word2Vec/FastText (T6) and MLM pretraining (T8). Openly licensed options, largest
first: **B-CORE** (`nahid-hub/B-CORE-bengali-corpus`, 4.32B tokens, quality-filtered and deduplicated) ·
**IndicCorp v2** bn (936M) · **CC-100** bn (8.3 GB) · **OSCAR** bn (632M words) · Bangla Wikipedia · the two
NCTB corpora.

**The stopping point is not 4B tokens.** Our encoder is ~25M parameters with a 6-hour budget on one 8.6 GB
laptop GPU. At roughly `6 × N` FLOPs per token, that budget consumes 0.5–1B tokens, and Chinchilla-optimal
training for 25M parameters is ≈20 tokens/parameter ≈ **500M tokens**.

> **Target ≈500M tokens** from B-CORE, with Wikipedia and both NCTB corpora mixed in and **up-weighted ×3** for
> domain relevance. Beyond that, extra text does not improve *this* model — the GPU binds, not the data. The
> report says exactly that rather than implying we were data-limited.

**Methodological requirement:** the tokenizer, Word2Vec, FastText and the MLM encoder all train on the
**identical frozen snapshot**. Otherwise RQ1 measures the data, not the representation.

### B. Retrieval index — richer makes the task *harder*, so it becomes an experiment

More passages means more distractors, so Recall@k **falls** as the index grows. A bigger index cannot flatter
the numbers. Rather than pick a size arbitrarily — which the instructor explicitly warned against — we measure
it: **nested indexes at 10k and 50k**, with a Recall@5-vs-size comparison per retriever. **Headline results are
reported at 50k**; we do not quote the 10k number because it flatters the system.

Composition at each size: **all 3,000 BanglaRQA gold contexts** (present at every size) + Wikipedia distractors
(*same distribution* as the gold — if every distractor were a textbook passage, a retriever could partly
succeed by learning "gold passages look like Wikipedia", inflating Recall@k through a distribution artefact) +
NCTB-SchoolText passages (the educational identity, the cited evidence in the demo, and realistic hard
distractors on curriculum topics).

### C. Supervised QA data — fixed by the benchmark

BanglaRQA's 14,889 pairs as released. Not enrichable without inventing labels.

*On Banglapedia:* good content, but copyrighted by the Asiatic Society with unclear reuse terms and orders of
magnitude smaller than B-CORE. Bangla Wikipedia already supplies encyclopedic curriculum content under CC BY-SA.

---

## 5. Answer equivalence — not punishing correct answers for wording

A correct answer phrased differently from the gold string is scored **wrong** by Exact Match. In Bangla this is
severe for three compounding reasons:

- **Inflection.** উদ্ভিদ / উদ্ভিদের / উদ্ভিদটি / উদ্ভিদগুলো are one lemma; token overlap sees four tokens.
- **Synonymy.** বায়ুমণ্ডল / আবহমণ্ডল, প্রক্রিয়া / পদ্ধতি — the textbook picks one, the student the other.
- **Script and transliteration.** CO₂ vs কার্বন ডাই-অক্সাইড; ASCII vs Bengali digits; সালোকসংশ্লেষণ vs সালোক সংশ্লেষণ.

**Our from-scratch FastText is the engine for this.** Subword n-grams make উদ্ভিদ and উদ্ভিদের near neighbours
by construction, so the model we had to train anyway is what solves it — which makes the "why FastText for a
morphologically rich language" argument measured rather than asserted.

### The four-tier ladder (T10)

| Tier | Metric | What it adds |
|---|---|---|
| 1 | **Strict EM** | The honest floor |
| 2 | **Token F1** (after Bangla normalisation) | **The headline number** — what the field reports |
| 3 | **Stem + synonym F1** | Suffix stripping and T6b synonym classes; removes the inflection and synonymy penalty |
| 4 | **Soft-embedding F1** + BERTScore | Greedy FastText alignment above a threshold, with BERTScore as an external reference |

**The gap between tier 1 and tier 4 is itself a result** — it quantifies how much apparent error is wording
rather than wrong answers.

### Two guards

**Tiers 3–4 are gated by the veto layer (§6).** A paraphrase and a contradiction look nearly identical to any
soft metric. Soft credit only applies once numerals, entities and polarity agree — applying synonym matching
*without* that gate makes evaluation worse, not better.

**No tier is quoted alone.** All four appear in every results table, the headline stays tier 2, and tier 3–4
thresholds are frozen on val before test is touched. Softer tiers are reported as upper bounds, not as the
system's score. *(Optional, if you ever have 30 spare minutes: hand-labelling 50 val predictions would let us
report which tier actually agrees with human judgement. The code path is left in place; nothing claims it
until the labels exist.)*

---

## 6. Verification and contradiction — the core contribution

The verifier is what separates this from a retrieve-and-read demo, and contradiction is its hard case.

### Why similarity alone fails

| Reference | Candidate | Embedding similarity | Truth |
|---|---|---|---|
| উদ্ভিদ কার্বন ডাই-অক্সাইড গ্রহণ করে | উদ্ভিদ **অক্সিজেন** গ্রহণ করে | very high | **wrong** |
| সালোকসংশ্লেষণ দিনে হয় | সালোকসংশ্লেষণ দিনে হয় **না** | very high | **wrong** |
| মুক্তিযুদ্ধ **১৯৭১** সালে | মুক্তিযুদ্ধ **১৯৫২** সালে | very high | **wrong** |

Every soft metric scores these ≈1.0. So verification runs the veto **before** any similarity signal.

### The veto layer (`verify/constraints.py`)

Extracts from both the candidate answer and the evidence: numerals and dates (Bengali *and* ASCII), units,
named entities, polarity (না / নয় / নেই and verb negation), and **directional relation order** — "কার্বন
ডাই-অক্সাইড থেকে অক্সিজেন" and the reverse have an identical token set and opposite truth. A contradiction on
any of these **caps support regardless of similarity**, and carries a human-readable reason so the UI can say
*সাল ভুল* rather than showing a bare low number.

> **Interaction bug to prevent.** Fuzzy matching — edit distance, char n-grams, transliteration — **must never
> reach veto terms.** Tolerant matching over numerals or entities can map ১৯৫২ → ১৯৭১ or অক্সিজেন → অক্সাইড and
> silently destroy the one layer that catches contradictions. Veto runs on normalised-but-**unfuzzed** text;
> fuzzy matching is confined to non-veto tokens. A unit test asserts this, because the failure mode looks like
> a *good* score.

### Training and measuring it

S3's negatives include **minimally-contrasting** ones, built by swapping exactly one critical token of a gold
answer. Random negatives teach nothing about contradiction — the model must separate strings differing by one
word. A held-out **contrast set** of minimal pairs is scored separately; near-chance accuracy there means the
verifier is a similarity detector, and the report says so rather than hiding it in the aggregate.

**The veto-on vs veto-off ablation on the contrast set is the second half of the headline sentence.**

---

## 7. Honest deviations from the proposal (report §1 and §5)

1. **NCTB-QA → BanglaRQA.** Gold QA supervision is Wikipedia-derived, not textbook-derived. The educational
   character lives in the retrieval index and the demo, but headline EM/F1 are BanglaRQA numbers. The report
   will not claim NCTB-QA results.
2. **Corpus collection is by citation, not scraping.** More reproducible, not less — but §5 must say so
   plainly, credit every source with its license, and report NCTB-SchoolText's chunking rule as *theirs*.
   Citation granularity is grade–subject–chapter, which is exactly what the proposal's §9 example shows.
3. **Pretraining is compute-limited, not data-limited.** ~500M tokens, 25M parameters, 6 hours. Expect to lose
   to BanglaBERT. **That gap is a result, not a failure** — reported as measured, with no tuning toward a
   nicer number.
4. **Generative reader and student grading are extensions** (E1, E2), not core claims.

---

## 8. Repository layout

```
NLP_Project/
├─ PLAN.md  TASKS.md
├─ data/
│  ├─ raw/{nctb_schooltext, bangla_textbook, wiki, bcore, banglarqa}/   # gitignored
│  ├─ processed/passages.jsonl            # retrieval index unit
│  ├─ processed/qa_{train,val,test}.jsonl
│  ├─ processed/train_corpus.txt          # the ~500M-token frozen snapshot
│  └─ sources.csv                         # source, license, URL, sha256, counts
├─ src/bnqa/
│  ├─ config.py                           # every path + hyperparameter, one place
│  ├─ data/        banglarqa · nctb_text · wiki · bigcorpus · dedup · splits · audit
│  ├─ preprocess/  normalize · tokenize · stopwords · stem · sentences
│  ├─ embeddings/  spm_train · word2vec · fasttext · pooling · intrinsic_eval
│  ├─ encoder/     model · mlm_pretrain · dual_encoder · contrastive_train
│  ├─ retrieval/   base · tfidf · bm25 · static_dense · neural_dense · pretrained_dense · hybrid · index
│  ├─ reader/      span_model · train_reader · predict
│  ├─ verify/      constraints · signals · fusion · calibration · threshold
│  ├─ eval/        retrieval_metrics · qa_metrics · verify_metrics · bootstrap · error_analysis
│  └─ pipeline.py                         # BanglaQA facade used by CLI, app, notebooks
├─ notebooks/      01_corpus … 08_error_analysis
├─ app/streamlit_app.py
├─ scripts/        run_all.ps1 · make_report_assets.py
├─ reports/        figures/ · tables/ · results.json
└─ tests/
```

**The one interface that makes the comparison valid:** `retrieval/base.py` defines a single `Retriever`
protocol — `build(passages)` / `search(query, k)`. Every retriever implements it, and the CLI, the app and
every evaluation notebook consume only that protocol. Same for `reader/predict.py` and `verify/`, both reached
through `pipeline.py`.

---

## 9. Task breakdown

| # | Task | Depends on | Est. |
|---|---|---|---|
| T0 | Repo skeleton, deps, config, seeds | — | 30 m |
| T1 | BanglaRQA ingest + leakage gate | T0 | 1 h |
| T2 | NCTB text corpora ingest (no PDFs) | T0 | 30 m |
| T2b | Stream B-CORE → ~500M-token frozen snapshot | T0 | 1–2 h |
| T3 | Index assembly at 10k/50k + dedup + audit | T1, T2 | 1–2 h |
| T4 | Bangla preprocessing + our own SentencePiece tokenizer | T2b, T3 | 2 h |
| T5 | Sparse retrieval: TF-IDF (word + char) and our own BM25 | T4 | 2 h |
| T6 | Word2Vec + FastText from scratch + intrinsic eval | T4 | 3 h |
| T6b | Synonym / inflection / transliteration resources | T6 | 2 h |
| T7 | Static-dense + RRF hybrid + query expansion + size comparison | T5, T6b | 3 h |
| T8 | **Our transformer encoder: MLM pretraining** (GPU, time-boxed 6 h) | T4 | 6 h |
| T9 | Dual-encoder contrastive training + neural retrieval arm | T8, T5 | 3 h |
| T10 | Span reader + the four-tier metric ladder | T8, T1, T6b | 4 h |
| T11 | **Verifier: veto layer, 4 signals, fusion, calibration, contrast set** | T10, T9 | 5 h |
| T12 | End-to-end pipeline + Streamlit demo | T11 | 3 h |
| T13 | Full evaluation, ablations, bootstrap CIs, error analysis | T12 | 3 h |
| T14 | Report draft from `reports/` assets | T13 | 3 h |
| — | *extensions — only if T0–T14 are complete* | | |
| E1 | Generative reader (BanglaT5/mT5) vs extractive | T13 | 4 h |
| E2 | Student-answer grading tab | T13 | 6 h |

**Critical path:** T0 → T1/T2/T2b → T3 → T4 → T8 → T10 → T11 → T12 → T13 → T14.

**Run T8 first and overnight.** It is the 6-hour GPU job and the only task that cannot be rescued by working
faster later — starting it on day 1–2 leaves room for a second attempt if it does not converge. T5/T6/T6b/T7
run on CPU while it trains.

**Minimum viable submission** if the week goes badly: T0–T7 + T10 + T11 + T12 — the retrieval comparison, a
working reader, a verifier with S1+S2 plus the veto layer, and a live demo. That still defends the headline
sentence; the four-signal fusion and the ablation study degrade first, not the end-to-end system.

### T0 — Repo skeleton, dependencies, config
Package tree, `requirements.txt`, `.gitignore` (data and checkpoints excluded), `TASKS.md`, and
`src/bnqa/config.py` holding **every** path and hyperparameter so no constant is hard-coded in a script.
**Reproducibility:** one `set_seed()` seeding python/numpy/torch, called by every training entry point, plus a
config hash written into every artefact and every row of `reports/results.json`, so any number traces back to
the configuration that produced it.
**Done when:** `python -c "import bnqa"` works, `pytest tests/` passes, config hash is stable across runs.

### T1 — BanglaRQA ingest + leakage gate
`load_dataset("sartajekram/BanglaRQA")` → `qa_{train,val,test}.jsonl` with
`qid, question, answers[], answer_type, question_type, is_answerable, gold_passage_id`.
**Leakage gate:** assert no `passage_id` appears in two splits; if the official split is question-level,
re-split by `passage_id` and record it in `reports/tables/splits.csv`.
**Done when:** three JSONLs exist, counts match 11,912 / 1,484 / 1,493 (or the re-split is documented), and the
leakage test passes in `pytest`.

### T2 — NCTB text corpora ingest *(no PDFs, no OCR)*
`nctb_text.py` downloads **NCTB-SchoolText** (Mendeley `f3882ccczp`, CC BY 4.0), reads
`processed_chapters_<subject>/*.jsonl`, normalises to `pid, text, grade, subject, chapter_no, chapter_title,
source`, and subsets to grades 6–10 curriculum subjects. Also pulls `md-nishat-008/Bangla-TextBook` (MIT).
Sources logged to `data/sources.csv` with license, URL, sha256 and counts — which is what the instructor's
"describe the collection process" guidance actually asks for.
**Fallback if Mendeley needs a login:** Bangla-TextBook alone supplies textbook passages; we chunk them in T3
and note the reduced chapter metadata.
**Done when:** `sources.csv` is populated and normalised textbook passages are on disk.

### T2b — Large training corpus
`bigcorpus.py` streams **B-CORE** with `streaming=True` — never materialising 52 GB — applies a length/script
filter and stops at the **500M-token budget**. Mixes in Bangla Wikipedia (`wikimedia/wikipedia`, `20231101.bn`)
and both NCTB corpora **up-weighted ×3**. Writes one immutable `train_corpus.txt` plus token count and sha256.
**Fallback chain if B-CORE is gated:** IndicCorp v2 bn → OSCAR bn → CC-100 bn → Wikipedia only.
**Done when:** `train_corpus.txt` exists with a token count within 10% of 500M, and its hash is in `sources.csv`.

### T3 — Index assembly
Keep NCTB-SchoolText's own chunking (pedagogically coherent units, median 351 chars) rather than re-chunking —
better motivated than a fixed token window, and §5 credits their rule as theirs. Only fallback text needs our
chunker (120–180 tokens, 30-token overlap). `wiki.py` supplies same-distribution distractors. `dedup.py` does
MinHash/Jaccard near-duplicate removal across sources. Build **nested** index sets at 10k and 50k so the size
comparison is controlled (50k is a superset of 10k; all 3,000 gold contexts are in both). `audit.py` reports
passage-length distribution and character-validity over a 200-passage sample.
**Done when:** `passages.jsonl` + two index manifests exist, every gold context is traceable at both sizes, and
`reports/tables/corpus_audit.csv` is written.

### T4 — Preprocessing + our own tokenizer
`normalize.py`: NFC, zero-width joiner/non-joiner repair, dari `।` normalisation, Bengali↔ASCII digit mapping,
quote/dash standardisation. `stopwords.py` + rule-based suffix stripper applied **only to sparse models** —
dense models keep the full surface form; the asymmetry is deliberate and reported.
`spm_train.py`: our own SentencePiece unigram tokenizer, vocab 32k, on `train_corpus.txt`.
**Done when:** normalisation round-trip tests pass and the tokenizer encodes/decodes Bangla losslessly.

### T5 — Sparse retrieval (BM25 written by us)
`tfidf.py`: word 1–2 grams **and** a separate char 3–5 gram vectoriser (absorbs Bangla inflection without a
morphological analyser). `bm25.py`: our own Okapi BM25 over a scipy sparse matrix, `k1`/`b` tuned on **val**.
`base.py` defines the `Retriever` protocol; `index.py` persists and logs latency + index size.
**Done when:** BM25 matches a hand-worked example in `pytest`, and both sparse retrievers are in
`reports/tables/retrieval.csv`.

### T6 — Static embeddings from scratch
gensim skip-gram Word2Vec and FastText on `train_corpus.txt`: dim 300, window 5, min_count 5, negative 10,
5 epochs, all cores; FastText adds char n-grams 3–6. `intrinsic_eval.py` reports nearest neighbours of
curriculum terms (সালোকসংশ্লেষণ, মুক্তিযুদ্ধ, যুক্তিযুক্ত), a ~50-pair hand-built similarity probe set, and an
OOV-coverage-vs-corpus-size curve — the concrete evidence that training worked, and the answer to "explain the
embedding process, don't treat it as a black box".
**Done when:** both models are saved and `reports/tables/intrinsic_embeddings.csv` + the OOV curve exist.

### T6b — Synonym, inflection and transliteration resources
All derived from `train_corpus.txt` and the T6 models — no external lexicon, so this stays "from scratch":
- `resources/synonyms.tsv` — FastText nearest-neighbour candidates filtered by cosine and co-occurrence, then
  **manually reviewed** for the ~300 curriculum terms that matter. Neighbour lists are noisy — antonyms and
  co-hyponyms sit close in embedding space — so unreviewed pairs are flagged and reported separately.
- `resources/variants.tsv` — English↔Bangla technical terms, Bengali↔ASCII digits, hyphen/space/compound forms.
- `resources/suffixes.txt` — the Bangla suffix list behind stem-F1, validated on a held-out word list.
**Done when:** `reports/tables/synonym_probe.csv` reports lexicon precision on a hand-built probe set, and
`bnqa.eval.qa_metrics` loads all three.

### T7 — Static-dense, hybrid, and the index-size comparison
`static_dense.py`: IDF-weighted mean pooling, with SIF + first-principal-component removal as a variant.
`hybrid.py`: reciprocal rank fusion over sparse + dense rankings.
**Query expansion experiment:** expand sparse queries with the top-2 FastText neighbours of each content word
(T6b) and measure how much of the sparse-vs-dense gap it closes — a direct, cheap answer to RQ1 that reuses the
synonym resources rather than adding machinery.
**Done when:** `reports/tables/retrieval.csv` covers TF-IDF(word), TF-IDF(char), BM25, W2V, FastText and RRF
hybrid — Recall@1/5/10, MRR@10, nDCG@10, latency, index size — at both index sizes, tuned on **val only**.

### T8 — Our transformer encoder, MLM pretraining *(highest-risk task)*
`encoder/model.py`: pre-LN transformer written by us — 6 layers, d_model 384, 6 heads, FFN 1536, max_len 256,
learned positions, GELU (~25M params + embeddings); fits 8.6 GB comfortably.
`mlm_pretrain.py`: 15% masking (80/10/10), AdamW lr 3e-4, 2k warmup, bs 32 × grad-accum 4, bf16, over
`train_corpus.txt`. **Time-boxed to 6 hours, checkpointing every 30 minutes** so a usable checkpoint always
exists. Logs tokens-seen alongside loss, so §7's compute-vs-data claim is evidenced rather than asserted.
**Done when:** the MLM loss curve is in `reports/figures/` and masked-token predictions on held-out Bangla
sentences are visibly sensible.

### T9 — Neural retrieval arm
`contrastive_train.py`: dual-encoder, mean pooling, InfoNCE over (question, gold passage) from the **train**
split, in-batch negatives + 2 hard negatives per query mined from BM25 (T5). lr 2e-5, 3 epochs.
`pretrained_dense.py` adds `csebuetnlp/banglabert` and LaBSE as the **labelled reference arm**.
**Done when:** `retrieval.csv` gains `ours-scratch-encoder` and `pretrained-ref` rows at both index sizes.

### T10 — Span reader + metric ladder
`span_model.py`: start/end span head on our encoder + SQuAD-2.0-style null span anchored at `[CLS]`, plus a
3-way answer-type head (**yes/no · single span · multiple spans**) — BanglaRQA's `answer_type` makes this
necessary, not optional. Multiple spans → top-n non-overlapping spans scored against the answer list.
`train_reader.py` trains on answerable + unanswerable; the null-vs-span threshold is tuned on val.
`eval/qa_metrics.py` implements the **four-tier ladder** of §5 plus HasAns-F1 / NoAns-F1. A pretrained-BanglaBERT
reader is trained as the reference arm.
**Done when:** all four tiers plus HasAns-F1/NoAns-F1 for both readers are in `reports/tables/reader.csv`.

### T11 — Verification, veto layer, calibration
`verify/constraints.py` — the veto layer of §6, including the directional-relation check, each veto carrying a
reason string. Runs on normalised-but-**unfuzzed** text; `tests/test_veto_isolation.py` asserts fuzzy matching
cannot reach veto terms.

`verify/signals.py` — the four signals of proposal §7.6:
- **S1 reader confidence** — normalised span probability, null-vs-span margin.
- **S2 evidence grounding** — token containment + char-3gram overlap + FastText cosine between the answer and
  its evidence sentence, matched through T6b's synonym/suffix resources so a correct paraphrase is not judged
  ungrounded. **Gated by the veto layer** (§5, §6).
- **S3 support classifier** — cross-encoder `[Q; A; P]` on our scratch encoder, with negatives from **four**
  sources: unanswerable items, wrong-passage retrievals, BM25 hard distractors, and **minimally-contrasting**
  swaps. The last is what teaches contradiction.
- **S4 multi-passage consistency** — run the reader over top-k, measure agreement by normalised string and
  embedding similarity.

`fusion.py` logistic regression (gradient-boosted cross-check) fitted on **val**; `calibration.py` temperature /
Platt scaling; `threshold.py` picks one threshold maximising abstention-aware F1.
**Done when:** reliability diagram, risk–coverage curve, ECE, AUROC of confidence-vs-correctness, **and
contrast-set accuracy with the veto layer on vs off** are in `reports/`.

### T12 — Pipeline + demo
`pipeline.py` wires preprocess → retrieve → read → verify → abstain-or-answer, returning
`⟨answer, evidence sentence, citation, confidence, supported?, veto_reason?⟩`.
`app/streamlit_app.py`: Bangla question box, retriever and reader selectors, answer + confidence bar + top-3
passages with the supporting sentence highlighted and grade–subject–chapter citation, an explicit abstention
message below threshold, and the veto reason when one fired.
**Done when:** every acceptance case in §12 behaves correctly.

### T13 — Full evaluation
Ablations with each module removed in turn — **including the veto-on vs veto-off ablation on the contrast set**,
which is the direct evidence that the verifier is not just a similarity detector. `bootstrap.py` 95% CIs over
1,000 resamples and **paired** bootstrap significance tests between retrievers. `error_analysis.py` buckets
failures into *retrieval miss · right passage wrong span · over-abstention on an answerable question ·
hallucination that survived verification*, with representative Bangla examples for each. One K-Means +
silhouette figure over passage embeddings (the surviving unsupervised component).
`scripts/make_report_assets.py` regenerates every table and figure.
**Done when:** `reports/results.json` is complete and every figure/table regenerates from a clean run.

### T14 — Report draft
Written from `reports/` assets, in your lab-report convention: numbered sections, references last, earlier
proposal drafts never cited.
1 Introduction · 2 Problem Statement · 3 Objectives · 4 Related Work · 5 Dataset and Corpus Construction
(the four honest deviations and the corpus-budget reasoning) · 6 Methodology · 7 Experimental Setup · 8 Results
· 9 Ablation and Error Analysis · 10 Limitations and Future Work · 11 Conclusion · References.

### E1 — Generative reader *(extension)*
BanglaT5 / mT5 fine-tuned on (question, retrieved passage) → answer with an explicit refusal string for
unanswerable items, compared against the extractive reader on identical contexts.

### E2 — Student-answer grading tab *(extension)*
A thin UI over the T11 verifier: student-answer box, optional teacher reference / marking points, marks total.
Adds keyword criticality weighting (corpus IDF × question-overlap discount × answer-type prior, collapsed into
concepts via T6b) and four question-type rubrics — confirmation → polarity only; factoid → veto-dominant;
causal → coverage capped without a causal connective; list → set scoring. The veto layer, contrast set and
contrastive negatives it needs already exist from T11.
**Labels would be synthetic**, so this ships as a demonstration, never as a validated grading claim.

---

## 10. Evaluation protocol (non-negotiable)

- Every hyperparameter — top-k, BM25 `k1`/`b`, null threshold, fusion weights, abstention threshold, tier 3–4
  match thresholds — is selected on **val**. The **test split is scored once**, at the end.
- Headline numbers are reported at the **declared 50k index size** and at **tier 2 (token F1)**. Softer tiers
  and the 10k index appear beside them, never instead of them.
- `reports/results.json` is written by eval scripts only; no number is hand-entered into the report.
- Every row carries the config hash and seed that produced it.
- If the from-scratch models lose to the pretrained reference, **the report says so and analyses why** (tokens
  seen, parameter count, compute). No metric is re-tuned to close a gap after seeing test results.

## 11. Risks and fallbacks

| Risk | Fallback |
|---|---|
| B-CORE gated or slow to stream | IndicCorp v2 bn → OSCAR bn → CC-100 bn → Wikipedia only; token budget unchanged (T2b) |
| Mendeley NCTB-SchoolText needs a login | `md-nishat-008/Bangla-TextBook` (ungated, MIT); we chunk it ourselves (T2) |
| MLM pretraining doesn't converge in 6 h | Use the best checkpoint; if unusable, the scratch **static** embeddings still carry the from-scratch requirement and the deficit is reported |
| Word2Vec/FastText overrun the CPU budget | 3 epochs, or subsample to 250M tokens **for all models alike** so the comparison stays controlled |
| BanglaRQA split is question-level → passage leakage | Re-split by `passage_id`; the T1 gate catches it |
| FastText neighbours give noisy synonyms | Manual review of the ~300 curriculum terms; unreviewed pairs excluded from the headline metric (T6b) |
| Soft metrics inflate scores | Four tiers always reported together; headline stays tier 2; tiers 3–4 gated by the veto layer |
| T8–T11 overrun | Fall back to the minimum viable submission (§9) |

## 12. Acceptance — how we know it works

1. `pytest tests/` — normalisation round-trips, chunker boundaries, BM25 against a hand-worked example, EM/F1
   against hand-computed Bangla cases, the split-leakage assertion, and `test_veto_isolation.py` proving fuzzy
   matching cannot reach veto terms (১৯৫২ never matches ১৯৭১; অক্সিজেন never matches অক্সাইড).
2. `python -m bnqa.data.audit` — passage-length distribution and character-validity over 200 passages.
3. `python -m bnqa.embeddings.intrinsic_eval` — nearest neighbours of curriculum terms are sensible.
4. `scripts/run_all.ps1` — corpus → index → train → evaluate → `results.json` and every figure, from clean.
5. `streamlit run app/streamlit_app.py`, three cases:
   - *"সালোকসংশ্লেষণ প্রক্রিয়ায় উদ্ভিদ কোন গ্যাস গ্রহণ করে?"* → cited answer, high confidence.
   - *"সালোকসংশ্লেষণে ব্যবহৃত এনজাইমের নাম কী?"* → **abstains** (the passage is topically similar but never names
     the enzyme) while still showing the closest passage.
   - A candidate answer contradicting the evidence (অক্সিজেন where the passage says কার্বন ডাই-অক্সাইড) →
     **rejected with a stated reason**, not accepted on similarity. This case proves the veto layer works; if it
     passes, the verifier is broken.

## 13. Assumptions

- The GPU is free for a ~6-hour run early in the week (T8), and ~30 GB of disk is free for the streamed corpus
  and checkpoints.
- Code and results first; the report (T14) is drafted once the numbers exist.
- No HuggingFace token needed — every dataset and reference model is public and ungated.

## 14. Sources

- NCTB-QA (unavailable): https://arxiv.org/abs/2603.05462
- BanglaRQA: https://huggingface.co/datasets/sartajekram/BanglaRQA · https://aclanthology.org/2022.findings-emnlp.186/
- NCTB-SchoolText (CC BY 4.0): https://data.mendeley.com/datasets/f3882ccczp/1
- Bangla-TextBook (MIT): https://huggingface.co/datasets/md-nishat-008/Bangla-TextBook
- B-CORE: https://huggingface.co/datasets/nahid-hub/B-CORE-bengali-corpus
- Bangla Wikipedia: https://huggingface.co/datasets/wikimedia/wikipedia (config `20231101.bn`)

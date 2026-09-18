# পাঠ-প্রমাণ (Path-Proman) — Evidence-Grounded Bangla Textbook QA with Verifiable Corpus Attribution

**CSE 4122 — NLP Lab Project · Final Build Plan v4 (the ultimate merge)**
*Merges `PLAN_v1_archive.md` (ML depth, 500M-token pretraining, per-task acceptance criteria),
`PLAN_v2_archive.md` (verifiability and showcase logistics) and `PLAN_v3_archive.md` (research
contributions). Nothing good from any of them is dropped — the **structure** changes, not the ambition.*

> **উত্তর নয় — প্রমাণসহ উত্তর.** Not an answer. An answer with proof.

A closed-domain Bangla question-answering system over NCTB textbook content that does four things a chatbot
cannot: it **retrieves** the evidence, **answers only from it**, **proves** the answer is a literal span of a
named corpus passage, and **abstains** — under a statistically guaranteed error bound — when the corpus does not
support an answer. It also **checks a candidate answer against the textbook**, naming the exact word that
contradicts the book.

Every output is a six-tuple:
**⟨answer, supporting sentence, corpus citation (grade→subject→chapter→pid→char-offsets), calibrated
confidence, supported? + veto reason, evidence receipt⟩**

> **Working agreement (unchanged since v1):** we build **one task at a time** — you say "do T3", I do T3 and
> stop. **Every v1 task ID keeps its v1 meaning.** Appendix A maps v1 → v4 so nothing you already planned
> around has moved.

---

## 0. The single structural decision in v4

### 0.1 Two tiers, joined by an interface

The v1 machine-learning stack — **the ~500M-token frozen corpus, our own SentencePiece tokenizer, Word2Vec and
FastText, the custom 6-layer ~25M-parameter transformer, and the complex multi-stage neural QA pipeline** — is
now **Tier B, an optional research extension**. We fully intend to build it (§8 specifies it to the
hyperparameter), but **nothing the project promises depends on it.**

| | **Tier A — the Guaranteed System** | **Tier B — the Research Extension** |
|---|---|---|
| What it is | Corpus · index · sparse + lexical retrieval · feature-based extractive reader · **the full verifier, veto layer and conformal abstention** · counterfactual attribution · interface · evaluation | **~500M-token frozen corpus** · **our SentencePiece 32k tokenizer** · **Word2Vec + FastText** · **our 6-layer ~25M-param transformer (MLM)** · bi-encoder, late-interaction and cross-encoder retrieval · neural multi-task reader · the pretraining-objective study, scaling curve, probing and distillation |
| Compute | **CPU only.** No GPU anywhere. | **RTX 3080 Laptop, 8.6 GB VRAM, ~24 GPU-hours** |
| Learned components | TF-IDF vectorisers, our BM25, question-type classifier, span ranker, support classifier, fusion model, calibrator, conformal threshold — **all fitted by us on our corpus** | Self-supervised pretraining from random init, plus every neural downstream head |
| Ships by | **19 Sep 21:00 (hard freeze)** | attempted from D0 night; whatever exists by 22 Sep is reported |
| If it fails | *it cannot* — there is no single point of failure that a GPU or a converging loss curve controls | the report states what did not converge and what it cost. **A negative result here is a publishable-honest finding, not a project failure.** |

**The joining principle, and the thing that makes this design good rather than merely safe:**

> **Every Tier-B component replaces exactly one Tier-A component behind an interface. Therefore Tier A is
> Tier B's ablation baseline.**

`retrieval/base.py` defines one `Retriever` protocol — `build(passages)` / `search(query, k)`. Tier A's BM25 and
Tier B's bi-encoder both implement it. `reader/base.py` defines one `Reader` protocol; Tier A's feature ranker
and Tier B's neural span model both implement it. Every verification signal is a function with a fixed
signature, so Tier B swaps the *implementation* of S2 and S3 without touching the fusion, the calibration, the
conformal threshold or the UI.

This buys three things at once: **(1)** the demo cannot be broken by a training run, **(2)** every Tier-B
component arrives with a built-in ablation for free, and **(3)** it creates the single most interesting research
question in the project — **RQ-A**, below.

### 0.2 What "from scratch" means at each tier

Sir's instruction was: *"if model used then pretrain it."* Both tiers comply, in different ways, and the report
says so explicitly.

| Tier | Compliance |
|---|---|
| **A** | **No pretrained model is used at all**, so the constraint is satisfied vacuously *and* substantively: every fitted component — TF-IDF, our Okapi BM25 implementation, the question-type classifier, the span ranker, the support classifier, the fusion model, the calibrator — is trained by us on our own corpus. Zero downloaded weights. |
| **B** | **Genuine self-supervised pretraining from random initialisation**: our tokenizer, our Word2Vec, our FastText, our transformer encoder on ~500M Bangla tokens. Pretrained BanglaBERT / LaBSE appear **only** as clearly-labelled reference and teacher arms. |

### 0.3 The v1 "Decisions taken" table, updated (v1 §2)

| Question | Decision |
|---|---|
| What "from scratch" means | §0.2. **Tier A: nothing downloaded. Tier B: our own tokenizer, Word2Vec, FastText, transformer encoder and span reader from random init.** Pretrained BanglaBERT/LaBSE appear **only as a clearly-labelled reference arm**. |
| Data | BanglaRQA supervised QA + NCTB-SchoolText passages + Bangla Wikipedia distractors + (Tier B) a large general Bangla corpus for training representations. |
| Deliverable | Git repo with an importable `src/bnqa` package; notebooks and the Streamlit demo both import it (**one code path**). |
| Scope | **Core = Tier A: the QA pipeline with verification and attribution. Tier B = the transformer stack, an optional research extension.** |
| Timeline | Showcase **20 Sep** and **23 Sep**. Tier A freezes 19 Sep 21:00; Tier B lands for the 23rd. |
| Grading | **Out of scope (v4).** v1 made student-answer grading an extension, v3 promoted it; v4 drops it. Its one load-bearing idea already lives in the verifier — see §12. |

### 0.4 Reading guide

If you only read four sections: **§2** (the tiers and RQ-A) · **§7** (what we build first) · **§11** (the
Verifiability Contract — the thing examiners will remember) · **§15** (the schedule and the freeze).

---

## Table of contents

| § | Section | § | Section |
|---|---|---|---|
| 0 | The single structural decision | 12 | Removed scope: student-answer grading |
| 1 | Purpose, goal, objectives | 13 | Repository layout |
| 2 | Contributions and research questions | 14 | Task breakdown (T · V · G · X · E) |
| 3 | Sir's requirements → where each is satisfied | 15 | Schedule |
| 4 | Context: four findings that changed the proposal | 16 | Evaluation protocol and ablation grid |
| 5 | Environment and compute | 17 | Risks and fallbacks |
| 6 | Corpus strategy | 18 | Acceptance and the live demo script |
| 7 | **Tier A methodology (the guaranteed system)** | 19 | Presentation outline |
| 8 | **Tier B methodology (the research extension)** | 20 | Division of labour |
| 9 | Answer equivalence — the four-tier ladder | 21 | Honest deviations |
| 10 | Verification, the veto layer, guaranteed abstention | 22 | Assumptions · 23 Sources |
| 11 | **The Verifiability Contract** | A–C | Appendices: v1→v4 map · hyperparameters · first commands |

---

## 1. Purpose, goal and objectives

### 1.1 Purpose

Bangla-medium students at NCTB grades 6–10 have no trustworthy digital way to ask a question about their
textbook and get an answer they can *check*. General chatbots answer fluently and confidently even when wrong,
and never point at a page. In education a confident wrong answer is worse than no answer: the student cannot
tell the difference, and neither can the teacher.

> **Purpose:** build a Bangla QA system whose every answer is auditable against a public textbook corpus, which
> refuses to answer when the corpus does not support one, and whose refusal rate is backed by a statistical
> guarantee rather than a hand-tuned threshold.

### 1.2 Goal

A working, offline Bangla QA system with a three-tab interface, where a teacher with no NLP
knowledge can verify any answer in under 30 seconds — plus a from-scratch transformer and a set of controlled
experiments that show what that transformer was worth.

### 1.3 Objectives

| # | Objective | Tier | Verified by |
|---|---|---|---|
| **O1** | Assemble a **known, public, licensed, hash-pinned** corpus of NCTB textbook passages plus same-distribution distractors, and document the collection process. | A | `data/sources.csv`, `reports/tables/corpus_audit.csv` |
| **O2** | Compare retrieval paradigms on **one index** under identical conditions, at two index sizes. | A (+B) | `reports/tables/retrieval.csv` |
| **O3** | Build an **extractive** reader whose answer is always a literal character span of a corpus passage. | A (+B) | `tests/test_span_is_substring.py` |
| **O4** | Build a **verifier** that separates supported from unsupported answers, catches **contradictions** that similarity accepts, and abstains under a **conformal selective-risk guarantee**. | **A** | contrast-set accuracy, veto ablation, `reports/tables/conformal.csv` |
| **O5** | Prove the system reads the corpus rather than memorising, via the **counterfactual corpus test**. | **A** | `reports/tables/counterfactual.csv`, live toggle |
| **O6** | Ship an interface where input → output, with evidence, citation, confidence and abstention visible. | **A** | §18 acceptance run |
| **O7** | Train **from random initialisation** a 32k SentencePiece tokenizer, Word2Vec, FastText and a **6-layer ~25M-parameter transformer** on ~500M tokens. | **B** | `MODEL_CARD.md`, `reports/figures/mlm_loss.png`, `reports/env.json` |
| **O8** | Run a **controlled pretraining-objective study** and explain the winner with layer-wise probes. | B | `reports/tables/objective_study.csv`, `probing.csv` |
| **O9** | Quantify **what the GPU bought** — every Tier-B component against its Tier-A baseline, with GPU-hours spent. | A×B | `reports/tables/tier_delta.csv` |
| **O10** | Report every result with val/test discipline, bootstrap CIs, and a written account of what failed. | A | `reports/results.json`, report §8–§10 |

---

## 2. Contributions and research questions

### 2.1 The one sentence this project has to defend (v1 §1, preserved)

> **Adding evidence-based verification with conformally-calibrated abstention cuts unsupported answers from X% to
> Y% at Z% coverage — and the same mechanism rejects contradictory answers that similarity-based verification
> accepts, while a counterfactual corpus swap shows the system's answers follow the corpus, not model memory.**

v1 noted this claim "holds regardless of how well our from-scratch models perform in absolute terms, because it
measures the *delta* from adding verification." **In v4 that is no longer a hedge — it is the architecture.**
The headline contribution lives entirely in Tier A and touches no GPU.

### 2.2 Five named contributions

- **C1 — Veto-gated, conformally-guaranteed verification.** *(Tier A · the headline.)* A constraint **veto
  layer** (numerals, dates, units, entities, polarity, directional relation order) runs **before** any
  similarity signal, because a paraphrase and a contradiction look identical to every soft metric. Five signals
  are fused, calibrated, and thresholded by **split-conformal selective-risk control**, turning "we picked a
  threshold" into *"with 90% confidence, at most 10% of answered questions are wrong."* Trained and measured on
  **BanglaVerify**, a 3-way support set auto-constructed from BanglaRQA with **minimal-pair** contradictions and
  a hand-verified held-out contrast set.

- **C2 — Counterfactual Corpus Attribution (CAR / AoRR) and the Verifiability Contract.** *(Tier A.)* A protocol
  that distinguishes a retrieval-grounded system from a memorising or generative one: perturb one fact in the
  corpus and measure whether the answer follows. Wrapped in a 14-point contract (§11) where every claim is
  checkable by someone who does not trust us.

- **C3 — MorphSpan-MLM and a controlled small-scale pretraining study for Bangla.** *(Tier B.)* Standard 15%
  random *subword* masking is close to degenerate in an agglutinative, inflectionally rich language: masking
  `ের` inside `উদ্ভিদের` is recoverable from the neighbouring subwords of the **same word**, so the model is
  rewarded for intra-word interpolation rather than context modelling. We mask **morphological units** instead —
  a whole stem-run or a whole suffix-chain — plus SpanBERT-style geometric spans over *words*. Then a
  **four-objective study at identical compute**, a **free tokens-seen scaling curve** from pinned checkpoints,
  a **domain-adaptive pretraining** stage, and **layer-wise morphological probes** that show *why* the winner
  wins.

- **C4 — A four-paradigm retrieval study on one index.** *(Tier A + B.)* Sparse (TF-IDF word, TF-IDF char 3–5,
  our own BM25) · single-vector dense (Word2Vec, FastText, our bi-encoder) · **late-interaction multi-vector**
  (ColBERT-style MaxSim) · cross-encoder reranking — plus ANCE-style iterative self-mined negatives and
  margin-MSE distillation. Every arm behind one protocol, so the comparison is not contaminated by plumbing.

### 2.3 Research questions

**RQ-A is new in v4, and it is the question the two-tier structure exists to answer.**

| # | Question | Tier | Safe if negative? |
|---|---|---|---|
| **RQ-A** | **What does ~24 GPU-hours of from-scratch pretraining actually buy over a well-engineered classical pipeline?** For every component — retrieval, reading, grounding, support classification — report the Tier-A score, the Tier-B score, the delta, and the GPU-hours spent. | A×B | **Yes, and this is the point.** "The transformer gained 4.1 F1 for 24 GPU-hours" and "the transformer gained nothing" are *both* good results, and both are honest answers to a question most student projects never ask. |
| **RQ1** | How do **sparse, single-vector dense, late-interaction and cross-encoder** retrieval compare on the same index at the same k, and how does the ranking change with index size (10k vs 50k)? | A+B | Yes — a finding either way is a finding. |
| **RQ2** | *(headline)* Does explicit evidence verification with a constraint **veto layer** reduce unsupported answers more than confidence thresholding alone, and does it catch **contradictions** that embedding similarity scores as near-identical? | **A** | Yes — it measures a delta. |
| **RQ3** | Under a **counterfactual corpus swap**, what fraction of answers follow the edited corpus (**CAR**), and does removing the evidence induce abstention (**AoRR**)? | **A** | A low CAR would be a major finding about our own system — and we would report it. |
| **RQ4** | **From scratch vs pretrained.** Our ~25M-parameter encoder on ~500M tokens against BanglaBERT's 110M on 2.5B — and **how much of the gap closes via task-level distillation without more data?** | B | Yes — expected to lose; the measured gap *is* the result. |
| **RQ5** | **Generative vs discriminative** (Sir's Q5, answered with a number): Naive Bayes vs Logistic Regression vs GBDT fusion on identical verifier features. | **A** | Yes |
| **RQ6** | At ~25M parameters and fixed compute, does **morphology-aware masking** beat random-subword and whole-word masking, and how does it compare to **ELECTRA-RTD**? Do layer-wise probes explain the difference? | B | Yes — a null result at 50M tokens is honest and we say so. |
| **RQ7** | How does downstream QA F1 scale with **tokens seen** (50M → 500M) at fixed parameters, and does **domain-adaptive pretraining** on NCTB textbooks add anything on top? | B | Yes — a flat curve says we were parameter-bound, not data-bound. |
| **RQ8** | Does **conformal selective-risk control** deliver its promised error bound on held-out test, and at what coverage cost versus a hand-tuned threshold? | **A** | Yes — the guarantee is distribution-free; *coverage* is the empirical question. |
| **RQ9** | Does the **four-tier answer-equivalence ladder** change the picture? How much apparent error is wording rather than wrong answers? | A(+B) | Yes — the tier-1-to-tier-4 gap is the result. |

---

## 3. Sir's requirements → where each one is satisfied

This table is **slide 2 of the presentation**. Nothing in his instructions is left unaddressed, and **every row
is satisfied by Tier A alone** — Tier B only exceeds it.

| What Sir said | How this plan satisfies it | Tier | Where |
|---|---|---|---|
| *"Interface thakte hobe, input dile output pabo"* | Streamlit app, four tabs, Bangla text box → answer + evidence + citation + confidence. No login, no setup. | A | §7.9, T12 |
| *"20, 23 তারিখ project showcase, presentation সহ"* | Two-deadline schedule with a **hard freeze on 19 Sep 21:00**; Tier A ships on the 20th, Tier B lands for the 23rd. | A/B | §15, §19 |
| *"corpus গরুর রচনা, means known corpus"* | Five **publicly known, citable, licensed** corpora, every file sha256-pinned and re-downloadable by the examiner. Nothing hand-made, nothing private. | A/B | §6, §23 |
| *"if model used then pretrain it"* | **Tier A uses no pretrained model at all and fits every component on our own corpus. Tier B adds genuine self-supervised pretraining from random init** — our tokenizer, Word2Vec, FastText and a 25M-parameter transformer on ~500M tokens. | A+B | §0.2, §8 |
| *"make the interface show the result"* | Every answer ships with the supporting sentence highlighted, exact pid and character offsets, the curriculum citation, a confidence bar with the conformal threshold marked, five signal bars, and the veto reason if one fired. | A | §7.9, §11 |
| **Q1** pretrained vs own embeddings | Tier B trains our own **and** runs the pretrained reference, **and** measures how much distillation closes the gap. His "balanced approach" taken literally and then some. | B | RQ4 |
| **Q2** corpus size | An explicit **computed** budget (§6.2) rather than an arbitrary number — plus an index-size experiment at 10k/50k **and** a tokens-seen scaling curve. Size is *measured* three ways, not assumed. | A+B | §6.2, RQ7 |
| **Q3** supervised vs unsupervised | Supervised core. Unsupervised/self-supervised components retained and reported: MLM pretraining, FastText synonym induction, K-Means topic clustering with silhouette, self-mined hard negatives. Not classification-bound, as he allowed. | A+B | §7.4, §8 |
| **Q4** GUI expectation | Deliberately plain Streamlit. Effort goes into *evidence display*, not chrome. | A | T12 |
| **Q5** generative vs discriminative | **RQ5** — Naive Bayes (generative) vs Logistic Regression and GBDT (discriminative) on identical verifier features, measured. E1 adds a generative reader as a labelled arm if there is ever time. | A | RQ5, E1 |
| **Q6** document the collection process | `data/sources.csv` (source, URL, license, sha256, counts, timestamp) + a full §5 in the report describing acquisition, filtering, dedup and chunking, crediting NCTB-SchoolText's chunking rule as **theirs**. | A | §6.3, §23 |

---

## 4. Context: the four findings that changed the proposal (v1 §2, preserved)

Two proposal PDFs in this folder describe a CSE 4122 (NLP Lab) project. Four findings from planning research
materially change it:

1. **NCTB-QA (arXiv:2603.05462), the proposal's primary dataset, cannot be obtained.** The paper states: *"The
   dataset used in this study is not publicly available as it is currently being utilized in ongoing research
   projects."* No GitHub / HuggingFace / Zenodo release exists.
2. **BanglaRQA is fully available** (`sartajekram/BanglaRQA`, CC-BY-NC-SA-4.0): 3,000 contexts, 14,889 QA pairs,
   split 11,912 / 1,484 / 1,493, with an `is_answerable` flag, four `question_type`s (confirmation, factoid,
   causal, list) and three `answer_type`s (yes/no, single span, multiple spans). **The answerable/unanswerable
   mix is what makes abstention trainable and measurable.** **Its contexts are Bangla Wikipedia, not NCTB
   textbooks** — a real deviation, handled in §21.1.
3. **NCTB textbook content is already published as clean text — no PDF work is needed.** The proposal's §7.1
   plans PDF download → PyMuPDF → Tesseract OCR fallback → chunking → audit. All of it is replaced by
   **NCTB-SchoolText** (Mendeley, CC BY 4.0) and **`md-nishat-008/Bangla-TextBook`** (HuggingFace, MIT). This
   removes the riskiest hours in the sprint and **an entire error class**: Bangla PDF extraction mangles
   conjunct glyphs through font-encoding bugs, and that noise would have propagated into every reported number.
4. **Openly licensed Bangla corpora exist at billion-token scale**, so Tier B's from-scratch models are limited
   by our GPU budget, not by available text (§6.2).

### 4.1 Verified on 17 Sep 2026 — implementation facts that are cheap to honour and expensive to rediscover

| Fact | Consequence |
|---|---|
| **BanglaRQA ships a loader script** (`BanglaRQA.py`); `datasets` v3+ refuses to execute dataset scripts. The payload is `Train.json` (27,818,012 B) · `Validation.json` (3,493,591 B) · `Test.json` (3,603,506 B). | **Download the raw JSON over HTTPS.** Do not depend on `load_dataset`. Removes a dependency and a failure mode. *(v1's T1 assumed `load_dataset` would work — it will not.)* |
| **BanglaRQA has no `answer_start` offsets.** Schema: `data[] → {passage_id, context, title, qas[] → {question_id, question_text, is_answerable, question_type, answers:{answer_text[], answer_type[]}}}`. | Span supervision requires aligning answer strings into contexts ourselves. **T1 must measure and report the alignment rate**; §17 has the fallback. **This is the most under-appreciated risk in the project.** |
| `passage_id` values look like `bn_wiki_2812`. | Confirms BanglaRQA contexts are Bangla Wikipedia. §21.1 stands. |
| **NCTB-SchoolText downloads with no login**: 11.6 MB zip, sha256 `10ce1f2d8b40b6cf600a067b9916e042a386c0e52d881dc518442b6376665b31`. | v1's "Mendeley may need a login" risk is **closed**. Pin that hash in `config.py`. |
| **Bangla-TextBook** is one CSV, `bangla_textbook_128w_cleaned.csv`, 188,650,371 B, **already chunked to 128 words**. | No chunker needed for it. |
| **Bangla Wikipedia** `20231101.bn` = 2 parquet shards, 183.5 + 144.5 MB. **B-CORE** = 10 parquet shards of **1.94 GB each**. | **Download 2 B-CORE shards (~3.9 GB) rather than streaming.** A downloaded, hashed subset is reproducible; a stream is not — and reproducibility is the spine of this project (VC-11). *(This revises v1's T2b, which planned `streaming=True`.)* |
| All five sources return HTTP 200 and are **ungated**. No HF token needed. | §23 |
| Windows console is cp1252 → `print()` of Bangla raises `UnicodeEncodeError`. | `PYTHONIOENCODING=utf-8` + `chcp 65001` in `scripts/env.ps1` (T0). |

### 4.2 The four v1 trims (v1 §3, preserved — each keeps the argument and drops machinery)

| Trim | Kept | Dropped |
|---|---|---|
| Index scaling | Two sizes, 10k and 50k — still empirical | Four sizes; halves the evaluation runs |
| Metric ladder | Four tiers (§9) | Three redundant middle tiers |
| Hybrid retrieval | Reciprocal rank fusion | Weighted score interpolation — RRF needs no score normalisation between incomparable scales |
| Student-answer grading | **Dropped in v4** (§12). v1's reasoning held up: the labels would have been synthetic, and the contradiction insight it was really about belongs in the verifier, where §10 now puts it | the grading deliverable, its rubrics, its human-agreement study |

---

## 5. Environment and compute

### 5.1 Two environments, both recorded

```
TIER A  (the guaranteed system — no GPU anywhere)
  CPU-only · Python 3.11+ · sklearn · scipy · numpy · pandas · pyarrow · streamlit · datasketch
  Verified present on the local machine: sklearn 1.8.0 · pandas 2.3.3 · numpy 2.4.4 · scipy 1.17.1
                                        pyarrow 25.0.1 · matplotlib 3.10.8 · seaborn 0.13.2
                                        tqdm 4.67.1 · streamlit 1.63.0   (Python 3.11.9)
  Tier A needs NO torch, NO transformers, NO gensim, NO sentencepiece.
  → This is why Tier A cannot be blocked by an install, a driver or a loss curve.

TIER B  (the research extension)
  RTX 3080 Laptop · 8.6 GB VRAM · CUDA · Ampere → bf16 native
  + torch (cu12x) · transformers · gensim · sentencepiece · bert-score
  ≥ 45 GB free disk · ≥ 16 GB RAM (FastText n-gram buckets are the RAM constraint)
```

> **Note on hardware.** I probed the machine this session and found no CUDA device (Intel Iris Xe, i5-1335U,
> 15.7 GB RAM). You have confirmed the training machine is a **separate RTX 3080 Laptop**. **T0 re-verifies on
> the training machine** and writes `reports/env.json` — so every compute claim in the report is evidenced on
> the box that actually produced it. **Tier A runs on either machine**, which is precisely why it is Tier A.

`reports/env.json` records: torch/CUDA versions, device name, total/free VRAM, CPU count, RAM, free disk, git
commit, config hash. **Every compute claim in the report cites this file.**

We deliberately **do not install `rank_bm25`** — BM25 is written by us (T5), which satisfies "from scratch" and
removes a dependency. We also **do not install FAISS**: brute-force cosine over a 50k × 300 float32 matrix is
60 MB and milliseconds, and three readable lines beat an opaque dependency.

### 5.2 Tier B's GPU budget — ≈24 hours across six days

| Job | GPU-h | When |
|---|---|---|
| Pre-tokenise ~500M tokens → `uint16` memmap *(CPU, blocking)* | — (0.5 h CPU) | D0 night |
| **X5 · Main pretrain: MorphSpan-MLM, ~500M tokens, time-boxed** | **6.0** | D0 night → D1 morning |
| X6 · Domain-adaptive continued pretraining on NCTB (~50M tokens) | 0.8 | D1 evening |
| X7 · Bi-encoder contrastive (InfoNCE + BM25 hard negatives) | 1.2 | D1 evening |
| X7b · Iterative self-mined negatives, round 2 (ANCE-lite) | 0.8 | D1 night |
| X10 · Cross-encoder support classifier on BanglaVerify | 1.2 | D1 night |
| X9 · Neural multi-task span reader (shared-norm + BIO), seed 1 | 1.2 | D2 morning |
| X8 · Late-interaction (MaxSim) reranker + cross-encoder rerank | 1.8 | D2 afternoon |
| X9b · Reader ensemble seeds 2–3 → S5 disagreement | 1.4 | D2 night |
| **— 20 Sep: SHOWCASE 1. GPU untouched. —** | 0 | D3 |
| **X11 · Objective study: 4 arms × 50M tokens at equal compute** | **5.0** | D4, unattended |
| X13 · BanglaBERT reference arms + task distillation | 2.5 | D4 night |
| X8b · Margin-MSE distillation cross-encoder → bi-encoder | 0.7 | D5 morning |
| X12 · Scaling-curve evals over 5 pinned checkpoints + probing | 1.7 | D5 |
| **Total** | **≈24.3** | |

**Two scheduling rules that matter more than anything else:**

1. **X5 launches on D0 night, before anything else is finished.** It is the only job that cannot be rescued by
   working faster later. All of Tier A runs on CPU *while it trains* — that is the whole point of the split.
2. **The objective study (X11) does NOT gate the main pretrain.** The production encoder uses MorphSpan-MLM on
   a priori grounds (C3's argument). X11 is a separate **equal-compute controlled comparison at 50M tokens**,
   run on D4 for the report — four arms, identical data order, identical seeds. If X11 ranks a different
   objective first, **we report that honestly**: the ranking at 50M need not hold at 500M, and we did not have
   the GPU-hours to check. Saying that costs nothing and buys a great deal of credibility.

---

## 6. Corpus strategy — "known corpus", documented and hash-pinned

### 6.1 The five corpora and what each is *for*

| Corpus | Size (verified) | License | Tier | Role |
|---|---|---|---|---|
| **BanglaRQA** `sartajekram/BanglaRQA` | Train 26.5 MB · Val 3.3 MB · Test 3.4 MB — 3,000 contexts, 14,889 QA pairs | CC BY-NC-SA 4.0 | **A** | The **only** labelled supervision. Also the substrate for **BanglaVerify**. |
| **NCTB-SchoolText** Mendeley `f3882ccczp` | **11.6 MB zip** — 58,872 chunks, 1,535 chapters, 34 subjects, classes 1–10, median chunk 351 chars | CC BY 4.0 | **A** | The **educational identity**. Carries grade / subject / chapter-no / chapter-title → this *is* our citation unit and the cited evidence in the demo. |
| **Bangla-TextBook** `md-nishat-008/Bangla-TextBook` | **180 MB CSV**, 87,110 rows, **already 128-word chunked** | MIT | **A** | 163 NCTB textbooks, grades 6–12. Second textbook source; the fallback if Mendeley fails. |
| **Bangla Wikipedia** `wikimedia/wikipedia` `20231101.bn` | **328 MB** (2 shards) → ~60M tokens | CC BY-SA 4.0 | **A** (distractors) / **B** (training) | (a) same-distribution distractors for the index; (b) encyclopedic content in the Tier-B training corpus. |
| **B-CORE** `nahid-hub/B-CORE-bengali-corpus` | 10 shards × **1.94 GB**; quality-filtered, deduplicated | CC BY 4.0 | **B** | The bulk of the ~500M-token pretraining corpus. **Download exactly 2 shards.** |

**Tier A's total download is ~550 MB. Tier B adds ~3.9 GB.** All five verified reachable and ungated.

*On Banglapedia:* good content, but copyrighted by the Asiatic Society with unclear reuse terms and orders of
magnitude smaller than B-CORE. Bangla Wikipedia already supplies encyclopedic curriculum content under CC BY-SA.
**Excluded on license grounds** — and saying so in the report is itself a point about responsible data collection.

### 6.2 "Rich" means three different things — conflating them wastes a week (v1 §4, preserved)

#### A. Representation-training corpus *(Tier B)* — richer is better, to a **computable** limit

Feeds the tokenizer (X2), Word2Vec/FastText (X3) and MLM pretraining (X5).

```
target  ≈500M tokens, frozen once, hashed once, never edited
  ≈430M  B-CORE (2 shards, filtered, sampled)
  ≈ 60M  Bangla Wikipedia 20231101.bn
  ≈ 10M  NCTB-SchoolText + Bangla-TextBook + BanglaRQA contexts, UP-WEIGHTED ×3  → ≈30M effective
```

> **The stopping point is not 4.32B tokens.** Our encoder is ~25M parameters with a 6-hour budget on one 8.6 GB
> laptop GPU. At roughly `6 × N` FLOPs per token that budget consumes 0.5–1B tokens, and Chinchilla-optimal
> training for 25M parameters is ≈20 tokens/parameter ≈ **500M tokens**. Beyond that, extra text does not
> improve *this* model — **the GPU binds, not the data.** The report says exactly that rather than implying we
> were data-limited, **and RQ7's scaling curve turns the claim into a figure** with either a visible plateau or
> a visibly unfinished slope. Both are honest; the second is more interesting.

**Step arithmetic, so the run is predictable:** batch 32 × grad-accum 4 × seq 256 = **32,768 tokens/step**;
500M tokens ⇒ **≈15,260 steps**; ≈4.5 TFLOPs/step at ~5 TFLOPS effective ⇒ **≈4–6 hours**. Checkpoint every
**1,000 steps**, plus **pinned evaluation checkpoints at 50M / 100M / 200M / 350M / 500M tokens seen** — which
is where RQ7's curve comes from **at zero extra training cost**.

**Methodological requirement (v1, preserved and now load-bearing for C3):** the tokenizer, Word2Vec, FastText,
the MLM encoder **and all four objective-study arms** train on the **identical frozen snapshot**
`data/processed/train_corpus.txt`, in the identical order, from the identical seed. Otherwise RQ1 and RQ6
measure the data, not the representation.

#### B. Retrieval index *(Tier A)* — richer makes the task *harder*, so it becomes an experiment

More passages means more distractors, so Recall@k **falls** as the index grows. **A bigger index cannot flatter
the numbers.** Rather than pick a size arbitrarily — which Sir explicitly warned against — we measure it:
**nested indexes at 10k and 50k** (50k is a strict superset of 10k; **all 3,000 BanglaRQA gold contexts are
present at both sizes**), with a Recall@5-vs-size comparison **per retriever**.

> **Headline retrieval numbers are reported at 50k.** We do not quote the 10k number on its own, because it
> flatters the system.

Composition at each size: all 3,000 gold contexts + Wikipedia distractors **drawn from the same distribution as
the gold** + NCTB passages.

> **Why the distractor distribution matters.** If every distractor were a textbook passage while every gold
> passage were Wikipedia, a retriever could partly succeed by learning *"gold passages look like Wikipedia"* —
> inflating Recall@k through a **distribution artefact** rather than relevance. Same-distribution distractors
> close that loophole. NCTB passages are added on top because they supply the educational identity, the demo
> citations, and realistic **hard** distractors on curriculum topics.

#### C. Supervised QA data *(Tier A)* — fixed by the benchmark

BanglaRQA's 14,889 pairs as released. **Not enrichable without inventing labels**, and inventing labels is
exactly where an examiner presses. The one place we *do* construct data — **BanglaVerify** (§10.3) — is built by
rule-based minimal-pair perturbation of *gold* answers, its construction is fully described, and a
**hand-verified sample reports per-generator label precision**. That is a different thing from inventing
supervision, and the report draws the distinction explicitly.

### 6.3 Collection process (Sir's Q6, answered concretely)

1. **Acquire** — `python -m bnqa.data.fetch_all` issues plain HTTPS GETs to the URLs in §23. No API key, no
   login, no scraping, no `datasets` script execution.
2. **Record** — every file's URL, license, download timestamp, byte size and **sha256** goes into
   `data/sources.csv`. NCTB-SchoolText's hash is **already known and pinned** (§4.1), so a mismatch is caught
   before parsing.
3. **Normalise** — every source maps to one schema:
   `{pid, text, source, grade?, subject?, chapter_no?, chapter_title?, title?}`.
4. **Filter** — drop passages under 40 characters or with <60% Bengali-script characters (catches boilerplate,
   English-only rows, markup residue). Counts logged per source.
5. **Deduplicate** — exact duplicates by normalised-text sha1, then MinHash/Jaccard near-duplicates at 0.85
   (`datasketch`), **across sources**. Counts before/after logged. *(Near-duplicate leakage between the index
   and the Tier-B training corpus would quietly inflate retrieval numbers, so this is a correctness step, not
   hygiene.)*
6. **Chunk** — **we do not re-chunk NCTB-SchoolText.** Its chunks are pedagogically coherent units (median 351
   chars) and its rule is credited as **theirs** in the report — better motivated than a fixed token window.
   Bangla-TextBook is already 128-word chunked. Only Wikipedia articles need our chunker (120–180 tokens,
   30-token overlap, sentence-boundary aligned).
7. **Audit** — `reports/tables/corpus_audit.csv`: per-source counts, length distribution, script validity over a
   200-passage sample, dedup removals, and the **answer-alignment rate** from T1.

---

## 7. Tier A methodology — the guaranteed system

Everything in this section runs on CPU, trains in minutes, and is explainable line by line. That constraint is
not a compromise: **it is why this project is more verifiable than a GPU-heavy one.** A logistic regression with
a printed feature-importance table is something a teacher can audit. A 25M-parameter black box is not.

### 7.1 Reproducibility spine (T0)

- `src/bnqa/config.py` holds **every** path and hyperparameter (Appendix B). No constant is hard-coded anywhere.
- `set_seed(seed)` seeds python / numpy / torch, called by **every** training entry point.
- A **config hash** (sha256 of the serialised config) goes into every artefact filename and every row of
  `reports/results.json`, so any number traces back to the configuration that produced it.
- `scripts/env.ps1` sets `PYTHONIOENCODING=utf-8`, `chcp 65001`, `HF_HOME` on the data drive.

### 7.2 Bangla normalisation and lexical preprocessing (T4)

`normalize.py`: NFC composition · **zero-width joiner / non-joiner repair** (ZWJ `\u200d`, ZWNJ `\u200c` — the
single most common source of "identical" Bangla strings that compare unequal) · dari `।` normalisation ·
**Bengali ↔ ASCII digit mapping** (০১২৩ ↔ 0123) · quote and dash standardisation · whitespace collapse.

`tokenize.py`: rule-based word tokenisation (whitespace + Bangla punctuation) and char n-gram extraction. **Tier
A needs no learned tokenizer** — TF-IDF and BM25 consume words and char n-grams directly.

`stopwords.py` + a rule-based suffix stripper, applied **only to sparse models** (v1, preserved). Dense models
keep the full surface form. **The asymmetry is deliberate and reported**: stemming helps lexical matching and
hurts distributional representations, and pretending otherwise would contaminate RQ1.

**Tests:** normalisation is **idempotent** (`norm(norm(x)) == norm(x)`), round-trips on a Bangla fixture set, and
**never alters the grapheme length of a veto term**.

### 7.3 Sparse and lexical retrieval (T5, T7)

- `tfidf.py` — word 1–2 grams, **and a separate char 3–5 gram vectoriser**. The char vectoriser absorbs Bangla
  inflection without a morphological analyser (উদ্ভিদ / উদ্ভিদের share most 3-grams). **Reporting both is
  itself an RQ1 result.**
- `bm25.py` — **our own Okapi BM25** over a scipy sparse matrix, `k1`/`b` tuned on **val**. **Validated in
  `pytest` against a hand-worked 3-document example.**
- `hybrid.py` — **Reciprocal Rank Fusion** (`1/(60 + rank)`) over the word, char and BM25 rankings.
- `index.py` — persists indexes and logs **latency and index size** alongside quality, because a retriever that
  is 40× slower for 1 point of Recall@5 is a different engineering proposition and the report should say so.
- **Query expansion experiment (v1, preserved):** expand sparse queries with the top-2 neighbours of each content
  word from the T6b resources, and measure how much of any gap it closes.

### 7.4 Derived lexical resources (T6b) — hand-authored in Tier A, induced in Tier B

Tier A cannot use FastText neighbours, so the resources are built the honest way:

- `resources/suffixes.txt` — the Bangla suffix list (ের · টি · গুলো · রা · কে · য় · তে · দের · খানা · গণ ·
  ভাবে · সমূহ · …), validated on a held-out word list. Behind stem-F1 **and** behind Tier B's MorphSpan-MLM.
- `resources/synonyms.tsv` — **hand-authored** for the ~300 curriculum terms that matter, plus candidates from
  **char-n-gram cosine** in the TF-IDF space. *(Tier B adds FastText-induced candidates and we report the
  precision of both — which makes "does distributional induction beat a hand list?" a measured question.)*
- `resources/variants.tsv` — English↔Bangla technical terms (CO₂ ↔ কার্বন ডাই-অক্সাইড), Bengali↔ASCII digits,
  hyphen/space/compound forms (সালোকসংশ্লেষণ ↔ সালোক সংশ্লেষণ).
- `reports/tables/synonym_probe.csv` reports **lexicon precision** on a hand-built probe set. Unreviewed pairs
  are flagged and **excluded from headline metrics**.
- **Unsupervised component (Sir's Q3):** K-Means + silhouette over TF-IDF passage vectors, one figure.

### 7.5 Question analysis (T10)

A rule table mapping Bangla interrogatives to an **expected answer class**. Every row is individually
defensible, which makes it excellent viva material:

| Question word | Expected class | Example |
|---|---|---|
| কে / কারা | PERSON | কে আবিষ্কার করেন? |
| কী / কি / কোনটি | THING / TERM | কোন গ্যাস গ্রহণ করে? |
| কোথায় | PLACE | কোথায় অবস্থিত? |
| কখন / কত সালে / কোন সালে | TIME / YEAR | কত সালে স্বাধীনতা লাভ করে? |
| কত / কয়টি / কতগুলো | NUMBER / QUANTITY | কতটি স্তর আছে? |
| কেন | REASON (clause) | কেন ঘটে? |
| কীভাবে / কিভাবে | PROCESS (clause) | কীভাবে হয়? |
| কি (sentence-initial) | POLARITY (yes/no) | উদ্ভিদ কি অক্সিজেন ত্যাগ করে? |

Plus a supervised **4-way `question_type`** classifier (confirmation / factoid / causal / list) trained on
BanglaRQA's own labels with the rule table as features. The predicted class **routes** decoding: POLARITY → the
verifier decides yes/no directly, LIST → multi-span set scoring, otherwise → single span.

### 7.6 The extractive span reader (T10) — feature-based, CPU, explainable

**Candidate generation.** For each top-k passage: split into sentences → score sentences by IDF-weighted
question overlap + char-n-gram cosine → take the top-3 → enumerate token n-grams of length 1–12 → filter by the
expected answer class (a TIME question keeps only spans containing a year-like token) → dedup. Typical yield:
200–600 candidates per question.

**Features (the table that goes in the report *and* the UI):**

| Group | Features |
|---|---|
| Lexical | IDF-weighted question-term overlap in the host sentence; fraction of question content words matched; min token-distance from span to nearest matched question term |
| Span-internal | length in tokens; mean/max token IDF; is-numeral; is-year; is-English-token; is-in-parentheses |
| Type match | expected-class match score (§7.5); answer-type prior given `question_type` |
| Similarity | char-3gram cosine(question, span); cosine(question, host sentence) *(Tier B swaps in FastText cosine — a free ablation)* |
| Structural | retriever rank + score of host passage; sentence rank; span position in sentence; relation-order flag |
| Negative | span equals a question term (usually wrong); span is a stopword-only n-gram |

**Model.** sklearn `LogisticRegression` and `HistGradientBoostingClassifier`, per-question softmax over
candidates, argmax wins. **Trains in under two minutes.** The best-vs-second margin and the absolute best score
become verifier features (S1) rather than a separate threshold.

**Answer-type head.** 3-way (yes/no · single span · multiple spans) — BanglaRQA's `answer_type` makes this
**necessary, not optional**. Multiple spans → top-n non-overlapping spans scored against the answer list as a set.

> **The feature-importance table is a deliverable, not a by-product.** "Why did the system pick this span?" is
> answerable with a bar chart in the UI. And when Tier B's transformer arrives, **this is the baseline it has to
> beat** — which turns its win (or loss) into a number instead of an assumption. That is RQ-A.

### 7.7 Verification — see §10 (the headline contribution, entirely Tier A)

### 7.9 The interface (T12)

Four tabs. Plain, fast, Bangla-first. **Built on D2, not D5** — it is what the examiners actually see.

**Tab 1 — প্রশ্ন করুন (Ask).** Bangla question box · **retriever selector** (demo RQ1 live) · **Answer** in large
type · **confidence bar with the conformal threshold marked** · the supporting sentence highlighted inside its
passage · citation line `শ্রেণি ৮ › বিজ্ঞান › অধ্যায় ৪: সালোকসংশ্লেষণ › pid nctb_8_sci_ch4_017 › chars 412–431` ·
**five signal bars** · top-3 passages · an explicit abstention message **with the closest passage shown anyway** ·
the veto reason when one fired · a **"🔍 প্রমাণ দেখুন (Verify)"** button. *(Tier B adds a reader selector and the
MaxSim heatmap.)*

Tab 1 also carries the **উত্তর যাচাই** control: paste a *candidate* answer and the verifier reports whether the
corpus supports it, **naming the contradicting token** when the veto layer fires. That is C1 doing its own job —
the same discrimination the reader's own answers go through — and it is what §18.2 case 4 demonstrates.

**Tab 2 — কর্পাস অনুসন্ধান (Corpus Explorer).** Free-text search over `passages.jsonl`, filter by grade /
subject / chapter, view any passage raw, and **"এই অনুচ্ছেদ থেকে প্রশ্ন করুন"** — ask a question against a passage
**the examiner chose**. This tab exists purely so a teacher can audit us.

**Tab 3 — সিস্টেম তথ্য (System card).** Parameter counts, tokens seen, training wall-clock, GPU, corpus hashes,
config hash, seed, the conformal guarantee and its achieved risk, the counterfactual results, and the
**Tier-A-vs-Tier-B delta table**. The "we are not a GPT wrapper" tab.

---

## 8. Tier B methodology — the research extension (the v1 transformer stack, in full)

**Attempted from D0 night. Reported honestly whether or not it wins.** Each item names the Tier-A component it
replaces, which is also its ablation baseline.

### X1 — The ~500M-token frozen corpus *(new capability)*

`bigcorpus.py` **downloads 2 B-CORE parquet shards** (~3.9 GB), applies the §6.3 length/script filter, samples to
the **500M-token budget**, mixes in Bangla Wikipedia and both NCTB corpora **up-weighted ×3**, and writes one
immutable `train_corpus.txt` plus token count and sha256.
**Fallback chain if B-CORE fails:** IndicCorp v2 bn → OSCAR bn → CC-100 bn → Wikipedia only; **token budget
adjusted and reported as adjusted.**
**Done when:** `train_corpus.txt` exists with a token count within 10% of 500M and its hash is in `sources.csv`.

### X2 — Our SentencePiece tokenizer *(replaces: Tier A's rule-based word tokeniser)*

**SentencePiece unigram, vocab 32,000**, trained on the frozen snapshot with
`input_sentence_size=10_000_000, shuffle_input_sentence=True` (500M tokens will not fit in the trainer's RAM),
`character_coverage=0.9995`, byte-fallback on, explicit `[PAD] [UNK] [CLS] [SEP] [MASK]`.

**Side artefacts C3 depends on:** a **subword → word map** and a **stem/suffix split** per word, cached to
`data/processed/morph_map.npz`. MorphSpan-MLM cannot be implemented without these.
**Done when:** the tokenizer encodes/decodes Bangla **losslessly**, and `reports/tables/tokenizer.csv` reports
fertility (subwords per word) beside BanglaBERT's tokenizer — a cheap, concrete comparison that also motivates
the 32k choice.

### X3 — Word2Vec and FastText from scratch *(replaces: Tier A's char-n-gram similarity features)*

gensim skip-gram on `train_corpus.txt`: **dim 300, window 5, min_count 5, negative 10, sg=1, epochs 3,
workers = cores − 2**; FastText adds **char n-grams 3–6** with `bucket=1_000_000` (the default 2M buckets ×
300 dims × 4 B = 2.4 GB of RAM; 1M halves it).

**These run on CPU while the GPU pretrains.** Expected: W2V ~30–50 min, FastText ~1.5–2.5 h.

`intrinsic_eval.py`: nearest neighbours of curriculum terms (সালোকসংশ্লেষণ, মুক্তিযুদ্ধ, বায়ুমণ্ডল, গণতন্ত্র,
যুক্তিযুক্ত) · a ~50-pair hand-built similarity probe · an **OOV-coverage-vs-corpus-size curve** · an
**inflection-pair probe** (উদ্ভিদ / উদ্ভিদের / উদ্ভিদটি / উদ্ভিদগুলো).

> **Why FastText, measured rather than asserted.** Those four surface forms are one lemma but four tokens.
> FastText's subword n-grams make them near neighbours **by construction**. The OOV curve and the inflection
> probe quantify it — and this is the same observation that motivates **C3's MorphSpan-MLM**, so the
> static-embedding analysis and the pretraining contribution reinforce each other.

**Done when:** both models are saved and `reports/tables/intrinsic_embeddings.csv` + the OOV curve exist.

### X4 — Static-dense retrieval and metric tier 4 *(replaces: Tier A's lexical-only retrieval and 3-tier ladder)*

`static_dense.py` — IDF-weighted mean pooling, plus a SIF + first-principal-component-removal variant.
Feeds RRF hybrid alongside the sparse arms. Also enables **tier 4 soft-embedding F1** (§9).
**Done when:** `retrieval.csv` gains W2V and FastText rows at both index sizes, and the ladder has four tiers.

### X5 — Our transformer encoder, MLM pretraining *(the highest-risk task, and the flagship)*

`encoder/model.py` — **a pre-LN transformer written by us:**

| Component | Choice | Params |
|---|---|---|
| Layers | 6, pre-LN | — |
| `d_model` | 384 | — |
| Heads | 6 (head dim 64) | — |
| FFN | 1536 (4×), GELU | 1.18M/layer |
| Attention | `F.scaled_dot_product_attention` (flash/SDPA path) | 0.59M/layer |
| Positions | **RoPE** *(fallback: learned absolute, as in v1)*, `max_len` 256 | 0 |
| Embeddings | 32,000 × 384, **tied** with the MLM head | 12.29M |
| Dropout | 0.1 | — |
| Init | `N(0, 0.02)`; residual output projections scaled by `1/√(2·n_layers)` | — |
| **Total** | | **≈23.0M** |

**Training mechanics that make the 6-hour budget real** — each is the difference between a 6-hour run and a
20-hour run, so none is optional:

- **Pre-tokenise the whole corpus once** into a flat `uint16` memmap (32k vocab fits `uint16`; 500M tokens =
  1 GB). Training reads contiguous slices — no tokenisation in the data loader, and the data order is
  reproducible from a seed.
- **bf16 autocast** (Ampere native; no GradScaler, unlike fp16). **Fused AdamW**, `betas=(0.9, 0.98)`,
  `eps=1e-6`, `weight_decay=0.01`, **no decay on LayerNorm/bias**. **lr 3e-4**, 2,000-step linear warmup →
  cosine to 3e-5. Grad clip 1.0. **bs 32 × grad-accum 4.**
- `pin_memory` · `num_workers=4` · `persistent_workers` · `non_blocking` transfers.
- **Time-boxed to 6 hours, checkpointing every 1,000 steps**, so a usable checkpoint always exists. Logs
  **tokens-seen** alongside loss.
- `torch.compile` is **not** used — Windows support is uneven and a compile failure at 1 a.m. on D0 is not a
  risk worth 15% throughput. Noted in the report as an untaken optimisation.

**Masking — MorphSpan-MLM (C3).** 15% token budget, standard **80/10/10** replacement:
1. With `p_morph = 0.5`, pick a word and mask **either its entire stem-subword-run or its entire suffix-chain**
   — never a fragment of either. Masking the stem forces inferring the lemma from context; masking the suffix
   chain forces inferring *grammatical role* from context.
2. Otherwise **SpanBERT-style geometric span masking over words** (`p = 0.3`, mean ≈1.8 words, cap 4), with
   whole-word expansion so a span never starts or ends mid-word.
3. **Salience tilt** (`λ = 0.3`): sample the masked word with probability `(1 − λ) + λ · normalised IDF`,
   mildly over-masking content words — motivated by the downstream task, where the answer is almost always a
   high-IDF token.

**Done when:** the MLM loss curve is in `reports/figures/`, masked-token predictions on held-out Bangla
sentences are visibly sensible, and the five pinned checkpoints exist.

### X6 — Domain-adaptive pretraining (RQ7b)

Continue pretraining ~50M tokens on the **NCTB textbook subset only** (DAPT), then re-run downstream evaluation.
Ablation: `general-only` vs `general + DAPT`. Cost 0.8 GPU-h.
**Done when:** the DAPT delta is a row in `reports/tables/tier_delta.csv`.

### X7 — Bi-encoder contrastive retrieval *(replaces: Tier A's RRF hybrid as the primary retriever)*

`dual_encoder.py` — shared-weight encoder, mean pooling, **InfoNCE** over (question, gold passage) from the
**train** split, temperature 0.05, in-batch negatives + **2 BM25 hard negatives per query**, lr 2e-5, 3 epochs.

**X7b — iterative self-mined hard negatives (ANCE-lite).** After epoch 1, re-encode the index with the *current*
student and mine top-ranked non-gold passages as round-2 negatives. Ablation: `in-batch only` vs `+BM25 hard`
vs `+self-mined`.
**Done when:** `retrieval.csv` gains `ours-scratch-encoder` rows at both index sizes.

### X8 — Late interaction and cross-encoder reranking

`late_interaction.py` — ColBERT-style: project each token to 128 dims, score
`Σ_{i∈q} max_{j∈p} cos(q_i, p_j)`. A full multi-vector index over 50k passages would need ~10 GB, so it is a
**reranker over the top-100** — 100 × ~180 tokens × 128 dims × 4 B ≈ **9 MB per query**.

> **Late interaction pays twice.** A third retrieval paradigm for RQ1, **and** the token-level MaxSim matrix is a
> **heatmap showing which passage tokens matched which question tokens** — an interpretability artefact for the
> UI (VC-13) that a single pooled vector cannot produce.

`cross_encoder.py` — `[CLS] q [SEP] p [SEP]` on our encoder; most accurate, slowest; reranks the top-20.
**X8b — margin-MSE distillation:** cross-encoder **teacher** → bi-encoder **student**,
`L = MSE((s_pos − s_neg)_student, (s_pos − s_neg)_teacher)` over mined triples.
**Done when:** all four paradigms are rows in `retrieval.csv` with latency and index size.

### X9 — Neural multi-task span reader *(replaces: Tier A's feature-based reader)*

Heads on our pretrained encoder: **span** start/end + SQuAD-2.0-style null anchored at `[CLS]` · **answer-type**
(3-way) · **question-type** (4-way) · **answerability** (binary) · **evidence-sentence** relevance ·
**BIO multi-span** tagging for list answers.

- **Uncertainty-weighted multi-task loss** (Kendall et al.): `L = Σ (1/(2σ_i²))·L_i + log σ_i`, learned `σ_i`,
  so head weights are not hand-tuned. **Ablation: single-task vs multi-task.**
- **Shared normalisation across passages** (Clark & Gardner): train with the span softmax normalised over **all**
  retrieved passages jointly, not within each. Two real consequences: scores become **comparable across
  passages**, which is what multi-passage QA needs; and the model is explicitly trained to say "no span here"
  for retrieved-but-wrong passages, which is the same discrimination abstention needs. **Ablation: per-passage
  vs shared.**
- **X9b — three-seed ensemble.** Mean span probabilities; the **disagreement** becomes verification signal S5.

**Done when:** all four metric tiers plus HasAns-F1 / NoAns-F1 for the neural reader are in
`reports/tables/reader.csv`, beside the Tier-A feature reader and the BanglaBERT reference reader.

### X10 — Neural support classifier *(replaces: Tier A's GBDT support classifier)*

Cross-encoder `[Q; A; P]` on **our** encoder, trained on **BanglaVerify** (§10.3) with its four negative
sources. Slots into S3 behind the same signature, so fusion, calibration, the conformal threshold and the UI are
untouched.
**Done when:** contrast-set accuracy for the neural S3 sits beside the Tier-A S3 in `verify_metrics.csv`.

### X11 — The controlled objective study (RQ6)

Four arms, **identical everything except the objective**: same frozen corpus, same data order, same seed, same
**50M-token** budget (~1,526 steps), same architecture, same optimiser.

| Arm | Objective |
|---|---|
| **A1** | Random subword MLM, 15%, 80/10/10 *(the BERT baseline)* |
| **A2** | Whole-word masking |
| **A3** | **MorphSpan-MLM** (ours) |
| **A4** | **ELECTRA-style replaced-token detection** — generator 3 layers / d 256, discriminator = our 6-layer model, `L_MLM(gen) + 50 · L_RTD(disc)`. Included because RTD's entire selling point is **sample efficiency at small scale**, which is precisely our regime. |

**Evaluation, and this is the subtle part.** MLM loss and RTD accuracy are **not comparable across objectives**,
so arms are ranked on **downstream proxies** at fixed cost: (a) frozen-encoder retrieval Recall@5 on val at the
10k index; (b) a 1-epoch reader probe's token F1 on val. Reported **with bootstrap CIs**, one seed per arm.

> **Stated limitation, up front:** one seed at 50M tokens may not resolve small differences, and a ranking at
> 50M need not hold at 500M. We report the CIs, name the budget as the constraint, and **do not claim
> significance we did not measure.** An honest null result on RQ6 is a perfectly good outcome — C3 remains a
> motivated, implemented and *probed* design.

### X12 — Scaling curve and probing (RQ7a, C3's mechanism)

- **Scaling curve, free.** Evaluate the **five pinned checkpoints** (50M/100M/200M/350M/500M tokens seen) on
  downstream retrieval Recall@5 and reader token F1. Plot quality vs tokens seen at fixed parameters. **Zero
  extra training cost.**
- **Layer-wise morphological probing.** Train linear probes on **frozen** per-layer representations to predict
  **which suffix class a token carries** — labels derived from our own suffix list, so **no external annotation
  is needed**. Compare MorphSpan-MLM against the random-subword baseline across all 6 layers.
  > If MorphSpan-MLM helps downstream, the probe should show *why*: better-separated morphological information
  > in the middle layers. If it helps downstream but the probe is flat, **we say so** — that is a genuinely
  > interesting negative result about the mechanism, and reporting it is the difference between analysis and
  > decoration.
- **Anisotropy.** Mean pairwise cosine of random token pairs per layer, ours vs BanglaBERT; and the effect of
  whitening / first-PC removal on retrieval. Explains *why* IDF-weighted pooling and SIF help.

### X13 — Scratch vs pretrained, and distillation (RQ4)

`pretrained_dense.py` adds `csebuetnlp/banglabert` and LaBSE as the **clearly-labelled reference arm** for
retrieval, plus a BanglaBERT reader. Then **task-level distillation BanglaBERT → our encoder**: how much of the
2.5B-vs-500M-token gap closes **without more data**?
**Done when:** `reports/tables/scratch_vs_pretrained.csv` has scratch, distilled and pretrained rows with
parameter counts, tokens seen and GPU-hours.

---

## 9. Answer equivalence — not punishing correct answers for wording (v1 §5, preserved)

A correct answer phrased differently from the gold string is scored **wrong** by Exact Match. In Bangla this is
severe for three compounding reasons:

- **Inflection.** উদ্ভিদ / উদ্ভিদের / উদ্ভিদটি / উদ্ভিদগুলো are one lemma; token overlap sees four tokens.
- **Synonymy.** বায়ুমণ্ডল / আবহমণ্ডল, প্রক্রিয়া / পদ্ধতি — the textbook picks one, the student the other.
- **Script and transliteration.** CO₂ vs কার্বন ডাই-অক্সাইড; ASCII vs Bengali digits; সালোকসংশ্লেষণ vs সালোক সংশ্লেষণ.

| Tier | Metric | What it adds | Available in |
|---|---|---|---|
| 1 | **Strict EM** | the honest floor | A |
| 2 | **Token F1** after Bangla normalisation | **THE HEADLINE NUMBER** — what the field reports | A |
| 3 | **Stem + synonym F1** | rule-based suffix stripping + the T6b synonym classes; removes the inflection and synonymy penalty | A |
| 4 | **Soft-embedding F1** + BERTScore | greedy **FastText** alignment above a threshold, with BERTScore as an external reference | **B** (X4) |

**The gap between tier 1 and tier 4 is itself a result** (RQ9) — it quantifies how much apparent error is
*wording* rather than *wrong answers*. Tier 3 already delivers most of that story in Tier A; Tier B completes it.

### Two guards (v1, preserved — both load-bearing)

**Tiers 3–4 are gated by the veto layer (§10).** A paraphrase and a contradiction look nearly identical to any
soft metric. Soft credit only applies once numerals, entities and polarity agree — **applying synonym matching
*without* that gate makes evaluation worse, not better.**

**No tier is quoted alone.** All available tiers appear in every results table, the headline stays tier 2, and
tier 3–4 thresholds are frozen on val before test is touched. Softer tiers are reported as **upper bounds**, not
as the system's score. *(Optional, 30 minutes: hand-label 50 val predictions and report which tier actually
agrees with human judgement. The code path is left in place; nothing claims it until the labels exist.)*

---

## 10. Verification, the veto layer, and guaranteed abstention — the core contribution (Tier A)

The verifier is what separates this from a retrieve-and-read demo, and contradiction is its hard case.

### 10.1 Why similarity alone fails (v1 §6, preserved)

| Evidence says | Candidate answer | Embedding similarity | Truth |
|---|---|---|---|
| উদ্ভিদ কার্বন ডাই-অক্সাইড গ্রহণ করে | উদ্ভিদ **অক্সিজেন** গ্রহণ করে | very high | **wrong** |
| সালোকসংশ্লেষণ দিনে হয় | সালোকসংশ্লেষণ দিনে হয় **না** | very high | **wrong** |
| মুক্তিযুদ্ধ **১৯৭১** সালে | মুক্তিযুদ্ধ **১৯৫২** সালে | very high | **wrong** |

Every soft metric scores these ≈1.0. **So verification runs the veto before any similarity signal.**

### 10.2 The veto layer (`verify/constraints.py`)

Extracts from **both** the candidate answer and the evidence: numerals and dates (Bengali *and* ASCII), units,
named entities, polarity (না / নয় / নেই and verb negation), and **directional relation order** — "কার্বন
ডাই-অক্সাইড থেকে অক্সিজেন" and its reverse have an **identical token set** and opposite truth, so
bag-of-words similarity is blind to it **by construction**. A contradiction on any of these **caps support
regardless of similarity** and carries a **human-readable reason string**, so the UI can say *"সাল ভুল — বইয়ে
১৯৭১, উত্তরে ১৯৫২"* rather than showing a bare low number.

> **Interaction bug to prevent — and this is the one that looks like success.** Fuzzy matching (edit distance,
> char n-grams, transliteration) **must never reach veto terms.** Tolerant matching over numerals or entities can
> map ১৯৫২ → ১৯৭১ or অক্সিজেন → অক্সাইড and **silently destroy the one layer that catches contradictions**. The
> veto runs on **normalised-but-unfuzzed** text; fuzzy matching is confined to non-veto tokens.
> **`tests/test_veto_isolation.py` asserts this**, because the failure mode manifests as a *better* score.

### 10.3 BanglaVerify — an auto-constructed minimal-pair support dataset (Tier A artifact)

S3 needs negatives that teach **contradiction**, not topic mismatch. **Random negatives teach nothing about
contradiction — the model must separate strings differing by one word** (v1 §6). So we construct a 3-way dataset
from BanglaRQA train:

| Label | Construction | ~Size |
|---|---|---|
| **SUPPORTED** | (evidence sentence, answer statement) from aligned gold spans | ~12k |
| **CONTRADICTED** | **minimal-pair perturbation** — swap exactly one critical token of the gold answer with another of the same type drawn from the corpus. **Six rule-based generators**, one per veto category: numeral, date/year, unit, entity, polarity insertion/deletion, relation-order reversal. | ~12k |
| **NEUTRAL** | answer statement paired with a **BM25-retrieved topically-similar passage that does not contain the answer** | ~12k |

**Quality control, because this is constructed data and an examiner will press there:**
- **Hand-verify a 300-item stratified sample** (≈50 min) and report **label precision per generator** in
  `reports/tables/banglaverify_quality.csv`. Any generator below ~90% precision is fixed or dropped.
- **A 500-pair held-out contrast set**, hand-verified, **never used in training**, scored **separately**.
- **License:** derivative of BanglaRQA ⇒ inherits **CC BY-NC-SA 4.0**. Stated in the release note.

### 10.4 The five signals (`verify/signals.py`)

Each is a function with a fixed signature, so Tier B swaps implementations without touching anything downstream.

- **S1 reader confidence** — normalised span score, null-vs-span margin, best-vs-second margin, host-passage
  retriever score.
- **S2 evidence grounding** — token containment + char-3gram overlap + rule-based stem match between the answer
  and its evidence sentence, matched **through** T6b's synonym/suffix resources so a correct paraphrase is not
  judged ungrounded. **Gated by the veto layer.** *(Tier B adds FastText cosine.)*
- **S3 support classifier** — **Tier A:** `HistGradientBoostingClassifier` over lexical + veto + overlap features
  on BanglaVerify (trains in seconds, fully explainable). *(Tier B: cross-encoder on our encoder — X10.)*
  Negatives from **four** sources: unanswerable items, wrong-passage retrievals, BM25 hard distractors, and
  **minimally-contrasting swaps**. The last is what teaches contradiction.
- **S4 multi-passage consistency** — run the reader over top-k and measure agreement by normalised string and
  char-n-gram similarity. **Disagreement across independent passages is a strong unsupported signal.**
- **S5 ensemble disagreement** — **Tier A:** bagged span rankers over subsampled features/seeds.
  *(Tier B: 3-seed neural ensemble — X9b.)* Epistemic uncertainty, a different quantity from S1's aleatoric
  confidence and empirically complementary.

### 10.5 Fusion and calibration (RQ5)

`fusion.py` — **Naive Bayes (generative) vs Logistic Regression vs GBDT (discriminative)** on identical
features, fitted on **val** only. **This is RQ5, and it answers Sir's Q5 with a number instead of a paragraph.**
`calibration.py` — a small **calibration study**: uncalibrated vs temperature vs Platt vs isotonic, compared on
**ECE, ACE and a reliability diagram**.

### 10.6 Conformal selective-risk control (`verify/conformal.py`) — RQ8

Most QA systems pick an abstention threshold to make a curve look nice. We **control the risk with a
distribution-free guarantee** instead:

1. Nonconformity score `s(x) = 1 − calibrated_confidence(x)` on a **held-out calibration split carved from val**
   (never from test).
2. For each candidate threshold τ on a grid, compute the **empirical selective risk** — the error rate among
   *answered* questions — and a one-sided **Clopper–Pearson upper confidence bound** at level δ = 0.1.
3. Choose the τ that **maximises coverage** subject to `UCB(risk) ≤ α`, with α = 0.10.

**Guarantee:** with probability ≥ 90% over the calibration draw, the error rate among answered questions is
≤ 10%. *(Geifman & El-Yaniv's selective-risk framing; Angelopoulos & Bates for the conformal machinery.)*

Then **verify it on test, once**: report the **achieved** selective risk and coverage, and whether the bound
held. *"We guarantee ≤10% error on answered questions with 90% confidence; on test we achieved 8.3% at 71%
coverage"* is worth more than any single F1 number in this project.

**Abstention-policy comparison (RQ8b), four arms:** (a) reader confidence only · (b) fused signals ·
(c) fused + veto · (d) fused + veto + **conformal**. Risk–coverage curves with AUC.

> **Why this matters beyond the metric.** This is an *education* system. "We can promise a bounded error rate on
> the questions we choose to answer" is the correct engineering guarantee for a student-facing tool, and it is a
> guarantee no chatbot offers. It converts abstention from a heuristic into a contract.

### 10.7 Measuring it honestly (v1, preserved)

The held-out **contrast set** is scored **separately** from the aggregate. **Near-chance accuracy there means the
verifier is just a similarity detector, and the report says so** rather than hiding it inside an average.
**The veto-on vs veto-off ablation on the contrast set is the second half of the headline sentence.**

*Bonus:* **confirmation (yes/no) questions are answered by the verifier itself** — polarity agreement between the
proposition and the evidence *is* the yes/no decision. One mechanism, two jobs, and the report says so.

---

## 11. The Verifiability Contract — how a teacher *knows* it works

**Fourteen claims, each independently checkable by someone who does not trust us. Every one is Tier A.**

| # | Claim | How the examiner checks it | Cost |
|---|---|---|---|
| **VC-1** | **Extractive by construction.** The answer is *always* a literal character span of a corpus passage. | The UI prints `pid`, `char_start`, `char_end` and the assertion `passages[pid].text[start:end] == answer` with a ✅. Ctrl+F the corpus. **A `pytest` asserts it for every test prediction.** | free — it is the architecture |
| **VC-2** | **Curriculum citation.** Every answer cites grade → subject → chapter → chapter title → pid. | Open the NCTB book at that chapter. | free — metadata is in the corpus |
| **VC-3** | **Offline-recheckable evidence receipt.** Each answer exports `reports/receipts/<qid>.json`: question, answer, pid, offsets, **passage sha256**, evidence sentence, confidence, all five signals, config hash, seed, timestamp. | `python -m bnqa.verify_receipt <file>` re-reads `passages.jsonl`, re-hashes the passage, re-checks the substring, prints **PASS/FAIL**. Runs on *their* machine, without us. | 1 h |
| **VC-4** | **Corpus Explorer.** The examiner searches the corpus themselves and asks a question against a passage **they** picked. | Tab 3. | 1 h |
| **VC-5** | **Abstention proof.** Ask something outside the corpus (*"২০২৬ বিশ্বকাপ কে জিতেছে?"*) → the system **abstains** and states that no supporting evidence exists. A chatbot answers. | Live, in the demo. | free — it is RQ2 |
| **VC-6** | **Offline / air-gapped run.** `BNQA_OFFLINE=1` installs a socket guard that raises on any outbound connection; the demo runs identically. | **Unplug the Wi-Fi in front of them** and run the demo. | 30 min |
| **VC-7** | **🏆 Counterfactual corpus test.** Swap a fact in the corpus; the answer follows the corpus. | Sidebar toggle **কর্পাস: আসল / পরিবর্তিত**. Ask *"মুক্তিযুদ্ধ কত সালে?"* → ১৯৭১. Flip it (one passage now says ১৯৬৯) → the answer becomes **১৯৬৯**, with the tampered sentence shown. Then *delete* the gold passage → it **abstains**. **A memorising or generative model cannot do this.** | 2 h — detailed below |
| **VC-8** | **Determinism receipt.** Same question + same seed + same config hash → byte-identical answer, pid and confidence. | Ask twice, compare. An API-backed model drifts. | free |
| **VC-9** | **Teacher audit sheet.** `reports/tables/audit_sample.csv` — 50 random test items with question, answer, gold, pid, chapter, offsets, confidence, supported?, verbatim evidence. | Spot-check any 10 rows in five minutes. **Hand this to the examiner on paper.** | 30 min |
| **VC-10** | **Model provenance card.** `MODEL_CARD.md` — parameter counts, tokens seen, wall-clock, loss curves, training timestamps, **checkpoint sha256**, `reports/env.json`, and an explicit statement of which tier used which weights (**no pretrained weights in either tier's main arm**). | Read it; check the hashes against the files. | 30 min |
| **VC-11** | **Reproducible from clean.** `scripts/run_all.ps1` regenerates corpus → index → models → evaluation → every table and figure. `data/sources.csv` has URL + license + sha256 for re-download. **This is why we download B-CORE shards instead of streaming.** | Run it. | in T13 |
| **VC-12** | **No hand-entered numbers.** `reports/results.json` is written **only** by evaluation scripts; the report reads from it. Every row carries config hash + seed. | Diff any number in the report against `results.json`. | free — discipline |
| **VC-13** | **Token-level alignment is visible.** *(Tier B)* The MaxSim heatmap shows which passage tokens matched which question tokens. | Expand the heatmap in Tab 1. | free — falls out of X8 |
| **VC-14** | **A stated error guarantee.** *"With 90% confidence, at most 10% of answered questions are wrong"* — distribution-free, plus the achieved risk on test. | `reports/tables/conformal.csv` and Tab 4. | in §10.6 |

### VC-7 in detail — the Counterfactual Corpus Test and the metric it yields

Cheap, the most persuasive thing in the project, and almost certainly unique in the showcase. It is a
**reportable metric**, not just a demo trick.

**Procedure.** Take *N* = 30 factoid questions the system answers correctly with high confidence. For each:
**(1) perturb** — in a *copy* of the gold passage, replace the answer string with a plausible same-type
alternative (১৯৭১→১৯৬৯; কার্বন ডাই-অক্সাইড→নাইট্রোজেন; a name → another name from the corpus); only that
passage changes and only that shard is rebuilt (seconds). **(2) re-ask** the identical question.
**(3) score** — did the answer follow the perturbed corpus?

**Corpus Attribution Rate (CAR)** = fraction of cases where the answer changes to the perturbed value. A
genuinely retrieval-grounded extractive system → **CAR ≈ 1.0**. A model answering from memorised parameters, or
any generative model, will not.

**Abstention-on-Removal Rate (AoRR)** = delete the gold passage entirely → the system should **abstain**, not
fabricate.

> This single experiment answers the examiner's real question — *"how do we know it isn't just a language model
> guessing?"* — with a number **and** a live demonstration, in 90 seconds.

---

## 12. Removed scope: student-answer grading — and what survives

v1 made student-answer grading an extension (E2). v3 promoted it to a Tier-A deliverable. **v4 removes it.**
The decision is recorded here rather than deleted quietly, because the reasoning behind it is the same reasoning
that justifies the verifier — and because "why isn't there a grading feature?" is a question worth being able to
answer in one sentence at the showcase.

### 12.1 Why it is out

v1 gave three reasons to keep grading at arm's length, and none of them got weaker:

* **It was not in the proposal the instructors approved.** Adding a second headline deliverable to a project
  already defending a verification claim splits the argument.
* **Its labels would have been synthetic.** Generated perturbations are not teacher judgements, and that is
  precisely where an examiner presses. The fix — a human-agreement study with three graders over 60 items —
  costs person-hours that the verifier, which the whole project rests on, needs more.
* **It cost ~7 hours plus integration**, in a schedule whose one protective rule is the 19 Sep freeze (§15).

Shipping a grading tab that we could not defend would have been worse than not shipping one.

### 12.2 What survives, and why nothing is actually lost

The load-bearing insight was never a *grading* feature. The verifier already asks *"is this answer supported by
this evidence?"*, which is the same discrimination as *"is this student's answer right?"* — so the parts that
carried the weight were already specified in §10 and stay exactly where they are:

* the **veto layer** — numerals, dates, units, entities, polarity, directional relation order (§10.2);
* the **contrast set** of minimal pairs, scored separately from the aggregate (§10.7);
* **minimally-contrasting negatives** for the S3 support classifier, via BanglaVerify (§10.3);
* the **fuzzy-matching isolation rule** and the test that enforces it (§10.2).

The demo also keeps its most persuasive moment. Tab 1's **উত্তর যাচাই** control takes a *candidate* answer and
reports whether the corpus supports it, naming the contradicting token when the veto fires — §18.2 case 4,
unchanged. That is C1 exercising its own mechanism on a user-supplied string, not a grading feature wearing a
disguise.

> The contradiction example an examiner remembers — *"বইয়ে আছে ১৯৭১, তুমি লিখেছ ১৯৫২"* — needed the veto layer,
> not a rubric. Removing grading costs the project a tab, not an argument.

### 12.3 If it ever comes back

Everything it would need already exists by then: the veto layer, BanglaVerify's minimal pairs, and the conformal
threshold that decides when to defer to a human. It is a UI over machinery being built anyway, which is why
dropping it now forecloses nothing. It stays out of scope for 20 and 23 September, and the report says so in
§21.8 rather than leaving a gap where a feature used to be.

## 13. Repository layout

```
NLP_Project/
├─ PLAN.md · TASKS.md · MODEL_CARD.md · README.md
├─ requirements-core.txt · requirements-ext.txt      # Tier A installs without torch
├─ .gitignore                                        # data/ and checkpoints/ excluded
├─ data/
│  ├─ raw/{banglarqa, nctb_schooltext, bangla_textbook, wiki, bcore}/      # gitignored
│  ├─ processed/passages.jsonl                       # THE retrieval + citation unit
│  ├─ processed/qa_{train,val,test}.jsonl
│  ├─ processed/banglaverify_{train,val,contrast}.jsonl
│  ├─ processed/index_{10k,50k}.manifest.json
│  ├─ processed/train_corpus.txt                     # [Tier B] frozen ~500M-token snapshot
│  ├─ processed/train_tokens.uint16.npy              # [Tier B] pre-tokenised memmap
│  ├─ processed/morph_map.npz                        # [Tier B] subword→word, stem/suffix
│  ├─ counterfactual/passages_tampered.jsonl         # VC-7
│  └─ sources.csv                                    # source · license · URL · sha256 · counts · timestamp
├─ src/bnqa/
│  ├─ config.py                                      # every path + hyperparameter (Appendix B)
│  ├─ data/        fetch_all · banglarqa · nctb_text · wiki · bigcorpus · dedup · splits · audit
│  ├─ preprocess/  normalize · tokenize · stopwords · stem · sentences · morph
│  ├─ embeddings/  spm_train · word2vec · fasttext · pooling · intrinsic_eval          # [Tier B]
│  ├─ encoder/     model · masking · mlm_pretrain · electra · dapt · probe             # [Tier B]
│  ├─ retrieval/   base · tfidf · bm25 · hybrid · index                                # Tier A
│  │               static_dense · dual_encoder · late_interaction · cross_encoder ·
│  │               pretrained_dense · distill                                          # [Tier B]
│  ├─ reader/      base · qtype · candidates · features · span_ranker                  # Tier A
│  │               span_model · multispan · predict                                    # [Tier B]
│  ├─ verify/      constraints · signals · banglaverify · fusion · calibration ·
│  │               conformal · threshold · receipt
│  ├─ eval/        retrieval_metrics · qa_metrics · verify_metrics · counterfactual ·
│  │               scaling · tier_delta · bootstrap · error_analysis
│  └─ pipeline.py                                    # BanglaQA facade used by CLI, app, notebooks
├─ notebooks/      01_corpus … 10_probing            # thin — they import src/, never redefine it
├─ app/streamlit_app.py
├─ scripts/        env.ps1 · run_all.ps1 · make_report_assets.py · verify_receipt.py
├─ reports/        figures/ · tables/ · receipts/ · results.json · env.json
└─ tests/
```

**The interfaces that make the whole design work** — `retrieval/base.py` (`Retriever`: `build` / `search`),
`reader/base.py` (`Reader`: `read(question, passages) -> Answer`), and the fixed signal signature in
`verify/signals.py`. Every arm of every tier implements them, and the CLI, the app and every evaluation notebook
consume **only** those protocols. **This is what makes Tier B a swap rather than a rewrite, and Tier A an
ablation baseline rather than a discarded draft.**

---

## 14. Task breakdown

**T = Tier-A core (v1 IDs preserved) · V = verifiability · X = Tier-B extension · E = post-deadline.**
⚡ = runs unattended. Appendix A maps v1 → v4.

### Tier A — must be complete by the 19 Sep 21:00 freeze

| # | Task | Depends on | Est. |
|---|---|---|---|
| **T0** | Repo skeleton · `requirements-core.txt` · `config.py` · seeds · UTF-8 console · `HF_HOME` · `reports/env.json` | — | 45 m |
| **T1** | BanglaRQA **raw-JSON** ingest · splits · leakage gate · **answer-alignment audit** | T0 | 1.5 h |
| **T2** | NCTB-SchoolText + Bangla-TextBook ingest · `sources.csv` with hashes | T0 | 45 m |
| **T3** | Index assembly 10k/50k nested · dedup · `corpus_audit.csv` | T1, T2 | 1.5 h |
| **T4** | Normalisation · rule tokeniser · stopwords · suffix stripper (sparse-only) | T3 | 1.5 h |
| **T5** | Sparse retrieval: TF-IDF word + char, **our BM25** | T4 | 2 h |
| **T6b** | Suffix list · hand-authored synonyms · variants · probe table | T4 | 1.5 h |
| **T7** | RRF hybrid · query expansion · **index-size comparison** · K-Means topic figure | T5, T6b | 2 h |
| **T10** | Question analysis · candidate generation · **feature span ranker** · metric ladder tiers 1–3 | T7, T1, T6b | 4 h |
| **T11** | **Verifier**: veto layer · S1–S5 · fusion (NB/LogReg/GBDT) · calibration · **conformal** · contrast set | T10 | 5 h |
| **T11b** | **BanglaVerify** construction + 300-item hand verification + GBDT S3 | T10, T3 | 3 h |
| **T12** | `pipeline.py` + **all three Streamlit tabs** | T11 | 3 h |
| **V1** | Evidence receipts · `verify_receipt.py` · `test_span_is_substring.py` | T12 | 1 h |
| **V2** | Offline mode (`BNQA_OFFLINE=1` socket guard) · determinism check | T12 | 45 m |
| **V3** | **Counterfactual corpus test** · CAR/AoRR · the sidebar toggle | T12 | 2 h |
| **V4** | Teacher audit sheet · `MODEL_CARD.md` | T13 | 1 h |
| **T13** | Evaluation pass → `results.json` → demo rehearsal | T12, V3 | 2 h |
| **🎯** | **SHOWCASE 1 — 20 Sep. Tier A complete.** | | |

**Tier A critical path:** T0 → T1/T2 → T3 → T4 → T5 → T7 → T10 → T11 → T12 → V1/V2/V3 → T13.
**Tier A total: ≈38 hours of work, no GPU, no torch.**

### Tier B — the research extension

| # | Task | Depends on | Est. |
|---|---|---|---|
| **X1** | B-CORE 2 shards + Wikipedia → **frozen `train_corpus.txt` (~500M tokens)** ⚡ | T0 | 2 h |
| **X2** | **SentencePiece 32k** · morph map · pre-tokenise → `uint16` memmap | X1, T6b | 2 h |
| **X3** | **Word2Vec + FastText** from scratch + intrinsic eval ⚡ *(CPU, runs during X5)* | X2 | 2.5 h ⚡ |
| **X4** | Static-dense retrieval + metric **tier 4** | X3, T7 | 1.5 h |
| **X5** | **Our transformer: MorphSpan-MLM, ~500M tokens, 6 h box, 5 pinned checkpoints** ⚡ | X2 | 6 h ⚡ |
| **X6** | **DAPT** on NCTB (~50M tokens) + ablation | X5 | 1 h |
| **X7** | **Bi-encoder** contrastive (InfoNCE + BM25 hard negatives) · **X7b** self-mined negatives | X5, T5 | 2.5 h |
| **X8** | **Late interaction (MaxSim)** + cross-encoder rerank · **X8b** margin-MSE distillation | X7 | 3 h |
| **X9** | **Neural multi-task span reader** (shared-norm, BIO) · **X9b** 3-seed ensemble → S5 | X5, T10 | 4 h |
| **X10** | Neural cross-encoder S3 on BanglaVerify | X5, T11b | 2 h |
| **X11** | **Objective study**: 4 arms × 50M tokens at equal compute + downstream probes ⚡ | X2, X5 | 5 h ⚡ |
| **X12** | **Scaling curve** over 5 checkpoints + **layer-wise morphological probing** + anisotropy | X5, X11 | 2 h |
| **X13** | **Scratch vs pretrained** (RQ4): BanglaBERT arms + **task distillation** | X7, X9 | 2.5 h |

### Closing tasks

| # | Task | Depends on | Est. |
|---|---|---|---|
| **T13b** | **`tier_delta.csv`** — RQ-A: every Tier-B component vs its Tier-A baseline, with GPU-hours | X4–X13 | 1.5 h |
| **T14** | Full evaluation: ablation grid · bootstrap CIs · error analysis · all figures | T13b | 3 h |
| **T15** | **Report draft** from `reports/` assets | T14 | 4 h |
| **T16** | Rehearsal · freeze · **backup demo recording** | T15 | 2 h |
| **🎯** | **SHOWCASE 2 / final submission — 23 Sep** | | |
| **E1** | Generative reader (BanglaT5/mT5) vs extractive, as a labelled arm | T14 | 4 h |
| **E2** | Auto quiz generation · chapter recommender | T14 | 4 h |

### Minimum viable submission (v1, preserved and now much safer)

**T0 → T7 + T10 + T11 (veto + S1 + S2) + T12 + V1 + V3 + T13.** The retrieval comparison, a working reader, a
veto-based verifier with abstention, a live demo, and the counterfactual test. **The headline sentence still
holds.** Degradation order: five-signal fusion → BanglaVerify's neural S3 → the ablation grid → the corpus-explorer tab.
**The end-to-end demo degrades last, never first.**

### Per-task acceptance criteria (v1's "Done when", preserved and extended)

**T0** — Package tree, `requirements-core.txt`, `.gitignore`, `TASKS.md`, and `config.py` holding **every** path
and hyperparameter so no constant is hard-coded in a script. `set_seed()` seeding python/numpy/torch called by
every training entry point; a config hash written into every artefact and every row of `results.json`.
**Done when:** `python -c "import bnqa"` works, `pytest tests/` passes, `reports/env.json` exists, and the
config hash is stable across runs.

**T1** — Raw-JSON download → `qa_{train,val,test}.jsonl` with
`qid, question, answers[], answer_type, question_type, is_answerable, gold_passage_id`.
**Leakage gate:** assert no `passage_id` appears in two splits; if the official split is question-level,
re-split by `passage_id` and record it in `reports/tables/splits.csv`. **Alignment audit:** report the fraction
of gold answers locatable in their context under the cascade exact → normalised → whitespace-insensitive → fuzzy.
**Done when:** three JSONLs exist, counts match 11,912 / 1,484 / 1,493 (or the re-split is documented), the
leakage test passes, and the **alignment rate is in `corpus_audit.csv`**.

**T2** — `nctb_text.py` downloads NCTB-SchoolText (hash-checked against the pinned sha256), reads
`processed_chapters_<subject>/*.jsonl`, normalises to `pid, text, grade, subject, chapter_no, chapter_title,
source`, and subsets to grades 6–10 curriculum subjects. Also pulls Bangla-TextBook.
**Done when:** `sources.csv` is populated with license, URL, sha256 and counts, and normalised textbook passages
are on disk.

**T3** — Keep NCTB-SchoolText's own chunking; chunk only Wikipedia (120–180 tokens, 30-token overlap).
`dedup.py` does MinHash/Jaccard removal across sources. Build **nested** index sets at 10k and 50k.
**Done when:** `passages.jsonl` + two index manifests exist, **every gold context is traceable at both sizes**,
and `reports/tables/corpus_audit.csv` is written.

**T4** — **Done when:** normalisation round-trip and idempotence tests pass, and the suffix stripper is applied
to sparse models only (asserted in a test).

**T5** — `base.py` defines the `Retriever` protocol; `index.py` persists and logs latency + index size.
**Done when:** **BM25 matches a hand-worked example in `pytest`**, and both sparse retrievers are in
`reports/tables/retrieval.csv`.

**T6b** — **Done when:** `reports/tables/synonym_probe.csv` reports lexicon precision on a hand-built probe set,
and `bnqa.eval.qa_metrics` loads all three resource files.

**T7** — **Done when:** `retrieval.csv` covers TF-IDF(word), TF-IDF(char), BM25 and RRF hybrid — Recall@1/5/10,
MRR@10, nDCG@10, latency, index size — **at both index sizes, tuned on val only**.

**T10** — **Done when:** metric-ladder tiers 1–3 plus HasAns-F1 / NoAns-F1 are in `reports/tables/reader.csv`,
and the **feature-importance table** is in `reports/tables/reader_features.csv`.

**T11** — **Done when:** reliability diagram, risk–coverage curve, ECE/ACE, AUROC of confidence-vs-correctness,
the conformal bound and its achieved test risk, **and contrast-set accuracy with the veto layer on vs off** are
all in `reports/`.

**T11b** — **Done when:** `banglaverify_quality.csv` reports per-generator label precision on the 300-item
sample, and the 500-pair contrast set is held out and never touched in training.

**T12** — `pipeline.py` wires preprocess → retrieve → read → verify → abstain-or-answer, returning
`⟨answer, evidence sentence, citation, confidence, supported?, veto_reason?, receipt⟩`.
**Done when:** every acceptance case in §18 behaves correctly.

**V1–V4, X1–X13** — acceptance criteria are stated inline in §8, §11 and §12.

**T13/T14** — **Done when:** `reports/results.json` is complete and **every figure and table regenerates from a
clean run** via `scripts/make_report_assets.py`.

**T15 — report structure (v1, preserved).** Written from `reports/` assets, in your lab-report convention:
numbered sections, references last, earlier proposal drafts never cited.
1 Introduction · 2 Problem Statement · 3 Objectives · 4 Related Work · 5 Dataset and Corpus Construction (the
honest deviations and the corpus-budget reasoning) · 6 Methodology (Tier A, then Tier B) · 7 Experimental Setup ·
8 Results (including **RQ-A: what the GPU bought**) · 9 Ablation and Error Analysis · 10 Limitations and Future
Work · 11 Conclusion · References.

---

## 15. Schedule (real dates)

Today is **Wed 17 Sep 2026**. Showcase **Sun 20** and **Wed 23**.

| Day | Date | Tier A (CPU) | Tier B (GPU) | Hard checkpoint |
|---|---|---|---|---|
| **D0** | **Wed 17, tonight** | T0 · T1 · T2 · T3 · T4 · T6b | **X1 · X2 → launch X5 (6 h) and X3 before sleeping** | **By 01:00: `passages.jsonl` exists AND X5 is running.** Disable Windows sleep first. If X5 is not launched tonight, Tier B slips a day. |
| **D1** | **Thu 18** | T5 · T7 · T10 (start) | collect X5/X3 → X4 · X6 · X7 · X7b · X10 ⚡ | **By 21:00: `retrieval.csv` complete at both index sizes.** RQ1 answered for Tier A. |
| **D2** | **Fri 19** | **T10 · T11 · T11b · T12 · V1 · V2 · V3 · T13** | X9 · X8 · X9b ⚡ | **21:00 — HARD FREEZE.** All six §18.2 demo cases pass end-to-end **on Tier A alone**. No new features after. Tag `showcase-1`. |
| **D3** | **Sat 20** | **🎯 SHOWCASE 1.** Rehearse twice; **record a backup screen capture.** | GPU untouched | Demo runs **offline** (VC-6) from a cold start in under 60 s. |
| **D4** | **Sun 21** | error analysis · audit sheet · rehearsal | **X11 (5 h, unattended)** · X13 ⚡ | X11's four arms complete. |
| **D5** | **Mon 22** | **T13b** · T14 · T15 · T16 | X8b · X12 | **`results.json` and `tier_delta.csv` complete; every figure regenerates from clean.** Tag `showcase-2`. |
| **D6** | **Tue 23** | **🎯 SHOWCASE 2 / final submission.** | | Everything frozen the night before. |

> **The one rule that saves this project: the 19 Sep 21:00 freeze, evaluated on Tier A alone.** If Tier B is
> working by then, wonderful — it slots in behind the interfaces. If it is not, the demo is unaffected. A working
> narrow demo on the 20th beats an ambitious broken one, and because Tier B lands on the 22nd, the "future work"
> slide will mostly be full by the 23rd anyway.

---

## 16. Evaluation protocol and ablation grid

### 16.1 Protocol (non-negotiable, v1 §10 preserved)

- **Every** hyperparameter — top-k, BM25 `k1`/`b`, InfoNCE temperature, null threshold, fusion weights,
  abstention threshold τ, conformal α and δ, tier 3–4 match thresholds — is selected on **val**.
  **The test split is scored once, at the end.**
- The conformal calibration split is carved from **val**, never from test.
- Headline numbers are reported at the **declared 50k index size** and at **tier 2 (token F1)**. Softer tiers and
  the 10k index appear **beside** them, never instead of them.
- `reports/results.json` is written **by evaluation scripts only**; no number is hand-entered into the report.
- Every row carries the **config hash and seed** that produced it.
- **Bootstrap 95% CIs** over 1,000 resamples; **paired** bootstrap significance tests between retrievers and
  between reader arms.
- **If the from-scratch models lose to the pretrained reference, the report says so and analyses why** — tokens
  seen, parameter count, compute. **No metric is re-tuned to close a gap after seeing test results.**
- **Tier A and Tier B are evaluated on identical splits, identical indexes and identical metrics**, or RQ-A is
  meaningless.

### 16.2 Metrics

| Stage | Metrics |
|---|---|
| Corpus | per-source counts · length distribution · script validity · dedup removals · **answer-alignment rate** · tokenizer fertility vs BanglaBERT |
| Retrieval | Recall@1/5/10 · MRR@10 · nDCG@10 · **latency** · index size — every arm, both index sizes |
| Reader | EM · **token F1 (headline)** · stem+synonym F1 · soft-embedding F1 · BERTScore · HasAns-F1 · NoAns-F1 · per-`question_type` · per-`answer_type` |
| Verifier | AUROC (confidence vs correctness) · **ECE + ACE** · reliability diagram · risk–coverage curve + AUC · **contrast-set accuracy, veto ON vs OFF** |
| Conformal | target α · δ · **achieved selective risk on test** · coverage · bound held? |
| Attribution | **CAR** · **AoRR** |
| Pretraining | MLM loss vs tokens-seen · **objective-study downstream proxies with CIs** · **scaling curve** · DAPT delta · probing accuracy per layer · anisotropy per layer |
| **RQ-A** | **per component: Tier-A score · Tier-B score · delta · CI on the delta · GPU-hours spent** |

### 16.3 The ablation grid (this table *is* report §9)

| Ablation | Removes | Tests |
|---|---|---|
| − veto layer | the constraint check | **RQ2 — the headline** |
| − S3 / − S4 / − S5 | one signal at a time | signal complementarity |
| − synonym resources | T6b | how much of the gain is lexical |
| − hybrid (single retriever) | RRF | RQ1 |
| − char n-grams | the char TF-IDF arm | Bangla inflection handling |
| conformal vs hand-tuned τ | the threshold rule | **RQ8** |
| NB vs LogReg vs GBDT fusion | the fusion family | **RQ5** |
| tier 1 → tier 4 | metric strictness | **RQ9** |
| **Tier A vs Tier B, per component** | the neural implementation | **RQ-A** |
| − shared normalisation | cross-passage softmax | X9 |
| − multi-task heads | auxiliary heads | X9 |
| − iterative negatives | ANCE round 2 | X7b |
| − distillation | margin-MSE | X8b |
| − DAPT | the textbook stage | RQ7b |
| MorphSpan vs WWM vs random vs RTD | the masking scheme | **RQ6** |
| scratch vs distilled vs BanglaBERT | the encoder | **RQ4** |

---

## 17. Risks and fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| **X5 does not converge in the 6-hour box** | medium | **Structurally harmless in v4.** Use the best pinned checkpoint; Tier A is unaffected; the deficit is a reported row in `tier_delta.csv`. *(In v1 this was the top risk. The tier split is precisely what retires it.)* |
| **BanglaRQA answers do not align into contexts** (no `answer_start`) | **medium — the one to watch** | T1 measures the rate. Cascade: exact → normalised → whitespace-insensitive → fuzzy (≤10% edit distance, **logged separately**). Unalignable items are excluded from span training and **the count is reported**. **If alignment < ~85%, pivot the reader to sentence-level answers** — VC-1 still holds, the span is just longer, and the metric ladder absorbs it. |
| **GPU unavailable** | low | Tier A ships regardless. Tier B becomes future work with the reasons stated. **Decide by 18 Sep 09:00, never mid-sprint.** |
| **8.6 GB VRAM OOM** | medium | bs 32 → 16 with grad-accum 8 (**identical effective batch**, so results stay comparable); seq 256 → 192; gradient checkpointing on the cross-encoder; BanglaBERT arms at bs 8. |
| B-CORE download slow (3.9 GB) | medium | One shard suffices for ~430M tokens at our filter rate. Fallback chain: **B-CORE → IndicCorp v2 bn → OSCAR bn → CC-100 bn → Wikipedia only**; token budget adjusted **and reported as adjusted**. |
| `load_dataset` refuses the script-based BanglaRQA repo | **closed** | Raw JSON over HTTPS; `datasets` is not on the critical path. |
| Mendeley NCTB-SchoolText fails | **closed** | Verified direct download, no login, 11.6 MB, sha256 pinned. Backup: Bangla-TextBook alone (ungated, already chunked). |
| BanglaRQA split is question-level → passage leakage | medium | **T1's leakage gate**; re-split by `passage_id` and record it in `splits.csv`. |
| **Near-duplicate leakage between the index and `train_corpus.txt`** | medium | MinHash dedup **across sources** (§6.3 step 5) and a reported overlap statistic. Quietly inflated retrieval numbers are worse than low ones. |
| MorphSpan-MLM needs the suffix list before X5 | **high if ignored** | **T6b runs on D0, before X2.** The suffix list is Tier A and hand-authored, so it exists early by design. **Never block the pretrain on a perfect lexicon.** |
| Word2Vec/FastText overrun the CPU budget | medium | 3 epochs; `bucket=1_000_000`; or subsample to 250M tokens **for all models alike** so the comparison stays controlled. |
| Objective study inconclusive at 50M tokens | **high** | **Expected.** Report CIs, name the budget limit, keep C3 as a motivated + implemented + probed design. A null RQ6 is an honest finding. |
| FastText neighbours give noisy synonyms | high | Manual review of the ~300 curriculum terms; unreviewed pairs excluded from headline metrics. **Tier A's hand list is the baseline, so this is measurable rather than fatal.** |
| Soft metrics inflate scores | medium | Four tiers always reported together; headline stays tier 2; tiers 3–4 **gated by the veto layer**. |
| BanglaVerify generators produce bad labels | medium | **300-item hand verification with per-generator precision**; any generator under ~90% is fixed or dropped. |
| Bangla text crashes the Windows console | **closed** | `PYTHONIOENCODING=utf-8` + `chcp 65001` (T0). |
| HF cache fills the system drive | medium | `HF_HOME` on the data drive, set in T0 **before any download**. |
| **Demo breaks on showcase day** | medium | **Backup screen recording made on D3 morning**, plus the frozen `showcase-1` tag. **Never demo from a branch edited that morning.** |
| Tier B work eats Tier A time | **high** | The freeze. The schedule assigns Tier B to GPU-idle hours and overnight slots. **If a Tier-B task threatens a Tier-A task, the Tier-B task dies.** |

---

## 18. Acceptance — how we know it works

### 18.1 Automated

1. `pytest tests/` — normalisation round-trips and **idempotence** · chunker boundaries · **BM25 against a
   hand-worked 3-document example** · EM/F1 against hand-computed Bangla cases · the **split-leakage
   assertion** · **`test_span_is_substring.py`** (every predicted answer is a literal substring of its cited
   passage) · **`test_veto_isolation.py`** (১৯৫২ never fuzzy-matches ১৯৭১; অক্সিজেন never matches অক্সাইড) ·
   suffix-stripper-is-sparse-only · *(Tier B)* tokenizer lossless round-trip and
   **`test_morph_masking.py`** (a mask never covers a partial stem or a partial suffix).
2. `python -m bnqa.data.audit` — length distribution, script validity, **answer-alignment rate**.
3. `python -m bnqa.eval.counterfactual` — **CAR and AoRR** → `reports/tables/counterfactual.csv`.
4. `python -m bnqa.verify.conformal --check` — the achieved selective risk respects the bound on test.
5. *(Tier B)* `python -m bnqa.embeddings.intrinsic_eval` — nearest neighbours of curriculum terms are sensible.
6. `scripts/run_all.ps1` — corpus → index → train → evaluate → `results.json` and every figure, **from clean**.

### 18.2 The live demo script (six cases, ~7 minutes) — rehearse this exactly

**All six pass on Tier A alone.** Tier B, if ready, only adds the reader selector and the heatmap.

| # | What we do | What must happen | What it proves |
|---|---|---|---|
| **1** | Ask *"সালোকসংশ্লেষণ প্রক্রিয়ায় উদ্ভিদ কোন গ্যাস গ্রহণ করে?"* | Correct answer, high confidence, evidence sentence highlighted, citation `শ্রেণি ৮ › বিজ্ঞান › অধ্যায় ৪`. | It works. |
| **2** | Click **প্রমাণ দেখুন** | pid, char offsets, the substring assertion ✅, and the `grep` command to find it in the corpus. *(Tier B: expand the MaxSim heatmap.)* | **VC-1, VC-3, VC-13** — the answer is *in* the corpus, verbatim. |
| **3** | Ask *"সালোকসংশ্লেষণে ব্যবহৃত এনজাইমের নাম কী?"* (topically close, never stated in the corpus) | **Abstains**, states that no supporting evidence exists, shows the closest passage anyway, and cites the **conformal guarantee**. | **VC-5, VC-14** — it knows what it doesn't know, with a stated error bound. A chatbot would invent an enzyme. |
| **4** | In Tab 1's **উত্তর যাচাই** box, paste a contradictory candidate answer (অক্সিজেন where the book says কার্বন ডাই-অক্সাইড) | **Rejected with a stated reason**, the contradicting word highlighted. **Not** accepted on similarity. | **C1** — the case that proves the veto layer. *If it passes, the verifier is broken.* |
| **5** | **Flip the sidebar to কর্পাস: পরিবর্তিত** and re-ask question 1 | The answer **changes to the tampered value**, with the tampered sentence shown. Flip back → correct again. | **🏆 VC-7** — the answer follows the corpus, not model memory. **This is the moment.** |
| **6** | Unplug the Wi-Fi and repeat question 1 | Identical answer, identical confidence, identical pid. | **VC-6, VC-8** — fully offline and deterministic. Not an API. |

**Then hand them `reports/tables/audit_sample.csv` on paper** (VC-9) and let them spot-check rows while we talk.

---

## 19. Presentation outline (16 slides, ~12 minutes)

1. **Title** — পাঠ-প্রমাণ · *"উত্তর নয় — প্রমাণসহ উত্তর"* · team · course.
2. **Sir's requirements → our answers** — the §3 table. Shows we listened, and frames everything that follows.
3. **The problem** — a chatbot answering a Bangla textbook question confidently and wrongly, with no citation.
   *"In education, a confident wrong answer is worse than no answer."*
4. **Four contributions (C1–C4)**, one line each — the slide that says "we did research, not a demo."
5. **The two-tier design** — the guaranteed system, the research extension, and the interface that joins them.
   *"Tier A is Tier B's ablation baseline."* This slide alone distinguishes the project methodologically.
6. **The corpus** — five known, public, licensed corpora; sizes; hashes; the collection process (Q6).
7. **What we trained from scratch** — Tier A: nothing downloaded, every component fitted by us. Tier B: our
   tokenizer, Word2Vec, FastText and **the 6-layer 25M-parameter transformer, ~500M tokens, 6 GPU-hours.**
8. **C1 — the veto layer** — the three-row contradiction table. Everyone understands ১৯৭১ vs ১৯৫২ instantly.
9. **C1 — the conformal guarantee** — risk–coverage curve, the bound, the achieved risk on test.
10. **C2 — 🏆 the counterfactual corpus test** — CAR and AoRR, then **the live toggle**.
11. **C4 — RQ1 retrieval** — four paradigms, both index sizes, with the honest finding.
12. **C3 — MorphSpan-MLM** — why random subword masking is degenerate in Bangla (the `উদ্ভিদের` example), the
    four-arm study, and the **probing** evidence.
13. **RQ7 — the scaling curve** — downstream F1 vs tokens seen. *"We were compute-limited, and here is the plot."*
14. **RQ-A — what did 24 GPU-hours buy?** The `tier_delta.csv` table. **The most interesting slide in the deck,
    whichever way the numbers fall.**
15. **Results and honest limitations** — including RQ4 (we lose to BanglaBERT, by how much, and why), the
    Wikipedia-not-textbook supervision deviation, and the measured index-composition check (§6.2B).
16. **Live demo** — the six cases from §18.2.

> **Slide 15's limitations are not a weakness; they are what make the other fifteen believable.** Examiners trust
> a project that reports its own gap far more than one that does not appear to have one.

---

## 20. Division of labour (4 people — adjust names)

| Role | Owns | Tasks |
|---|---|---|
| **A — Data & retrieval** | corpus, index, sparse retrieval, RQ1 | T1 T2 T3 T5 T7 · X1 X4 X7 X8 |
| **B — Pretraining & encoder** *(Tier B lead)* | tokenizer, morph map, W2V/FastText, MLM, DAPT, objective study, probing, `MODEL_CARD.md` | T4 T6b · X2 X3 X5 X6 X11 X12 |
| **C — Reader & verifier** | span reader, veto layer, signals, BanglaVerify, calibration, conformal | T10 T11 T11b · X9 X10 X13 |
| **D — Interface & verifiability** | Streamlit, receipts, offline mode, counterfactual test, audit sheet | T12 V1 V2 V3 V4 T13 |
| **All** | T13b, rehearsals, the report | T13b T14 T15 T16 |

**Everyone must be able to run the demo and explain the veto layer**, because the examiner picks who answers.
**Role B's work is Tier B** — which means if it runs late, nothing else is blocked. That is deliberate.

---

## 21. Honest deviations from the original proposal (report §1 and §5)

State these up front. **Pre-empting an objection is worth more than surviving it.**

1. **NCTB-QA → BanglaRQA.** The proposal's primary dataset is not obtainable — the paper states it *"is not
   publicly available as it is currently being utilized in ongoing research projects"*, and no GitHub /
   HuggingFace / Zenodo release exists. So gold QA supervision is **Wikipedia-derived, not textbook-derived**
   (confirmed by BanglaRQA's own `bn_wiki_*` passage ids). The educational character lives in the **retrieval
   index, the citations and the demo**; headline EM/F1 are BanglaRQA numbers. **The report
   will not claim NCTB-QA results.**
2. **Corpus collection is by citation, not scraping.** More reproducible, not less — but §6 says so plainly,
   credits every source with its license, and reports NCTB-SchoolText's chunking rule as **theirs**. Citation
   granularity is grade–subject–chapter, exactly what the proposal's §9 example shows.
3. **No PDF/OCR pipeline.** The proposal's §7.1 plan is replaced by two pre-cleaned published corpora, removing
   an entire error class: Bangla PDF extraction mangles conjunct glyphs through font-encoding bugs.
4. **The transformer is a research extension, not the core.** The proposal implied a neural pipeline throughout.
   In v4 the **headline contribution (verification, attribution, abstention) runs without it**, and the
   transformer is evaluated as an *intervention* against a classical baseline (**RQ-A**). This is a deliberate
   methodological choice, stated as such — not a retreat.
5. **Pretraining is compute-limited, not data-limited.** ~23–25M parameters, ~500M tokens, 6 GPU-hours on one
   8.6 GB laptop GPU, against BanglaBERT's 110M parameters and 2.5B tokens. **We expect to lose. That gap is a
   result, not a failure** — reported as measured, with no tuning toward a nicer number, and accompanied by
   RQ7's scaling curve and RQ4's distillation intervention.
6. **The objective study is under-powered by design.** One seed per arm at 50M tokens, because ~24 GPU-hours is
   the whole budget. We report CIs and do not claim significance we did not measure.
7. **BanglaVerify is constructed, not annotated.** Rule-based minimal-pair perturbation with a hand-verified
   300-item sample reporting per-generator label precision. A **derived resource with measured quality**, not
   human-annotated gold.
8. **Student-answer grading is out of scope** (§12). Dropped rather than shipped half-validated: its labels
   would have been synthetic, and the contradiction detection it was really about is the verifier's job.
9. **Generative reader (E1) is out of scope** for 20/23 Sep. Generation would weaken VC-1 (extractive by
   construction), so if it is ever added it stays a labelled comparison arm, never the main path.

---

## 22. Assumptions

- **Tier A assumes only a CPU and ~550 MB of downloads.** This is the assumption the project's promises rest on,
  and it is satisfied on the machine I can already see.
- **Tier B assumes** the RTX 3080 Laptop (8.6 GB VRAM, CUDA) is available for **~24 GPU-hours across six days**,
  including a **6-hour unattended run on the night of 17 Sep**. **Disable Windows sleep before launching it** —
  otherwise the overnight job stops silently and D1 begins with nothing. Also **≥ 45 GB free disk** and
  **≥ 16 GB RAM** (FastText's n-gram buckets are the RAM constraint) and **~4 GB** more of downloads.
- One person is available for ~50 minutes for BanglaVerify's 300-item hand verification (T11b).
- Code and results first; the report (T15) is drafted once the numbers exist.
- No HuggingFace token is needed — every dataset and reference model is public and ungated *(verified)*.

---

## 23. Sources — verified status, licenses and hashes

| Source | URL / file | License | Verified 17 Sep 2026 |
|---|---|---|---|
| **BanglaRQA** | `huggingface.co/datasets/sartajekram/BanglaRQA` → `Train.json` (27,818,012 B) · `Validation.json` (3,493,591 B) · `Test.json` (3,603,506 B) | CC BY-NC-SA 4.0 | ✅ 200, ungated. **Download raw JSON — do not use `load_dataset`.** **No `answer_start` field.** |
| **BanglaRQA paper** | `aclanthology.org/2022.findings-emnlp.186/` | — | cite in Related Work |
| **NCTB-SchoolText** | `data.mendeley.com/datasets/f3882ccczp/1` → `NCTB-SchoolText.zip` | CC BY 4.0 | ✅ **11.6 MB, no login.** sha256 `10ce1f2d8b40b6cf600a067b9916e042a386c0e52d881dc518442b6376665b31` — **pin in `config.py`** |
| **Bangla-TextBook** | `huggingface.co/datasets/md-nishat-008/Bangla-TextBook` → `bangla_textbook_128w_cleaned.csv` (188,650,371 B) | MIT | ✅ 200, ungated, **already 128-word chunked** |
| **Bangla Wikipedia** | `huggingface.co/datasets/wikimedia/wikipedia` → `20231101.bn/train-0000{0,1}-of-00002.parquet` (183.5 + 144.5 MB) | CC BY-SA 4.0 | ✅ 200, ungated |
| **B-CORE** | `huggingface.co/datasets/nahid-hub/B-CORE-bengali-corpus` → `data/train-0000{0,1}-of-00010.parquet` (**1.94 GB each**) | CC BY 4.0 | ✅ 200, ungated. **Download 2 shards, do not stream** (VC-11). |
| **BanglaBERT** *(reference / teacher arm only)* | `huggingface.co/csebuetnlp/banglabert` | — | ✅ 200, ungated |
| **NCTB-QA** *(unobtainable)* | `arxiv.org/abs/2603.05462` | — | ❌ not released — cite as unavailable, §21.1 |

**Methods cited in the report** — BERT (Devlin et al. 2019) · SpanBERT (Joshi et al. 2020) · ELECTRA (Clark et
al. 2020) · RoPE (Su et al. 2021) · Chinchilla (Hoffmann et al. 2022) · DAPT (Gururangan et al. 2020) · DPR
(Karpukhin et al. 2020) · ANCE (Xiong et al. 2021) · ColBERT (Khattab & Zaharia 2020) · margin-MSE distillation
(Hofstätter et al. 2020) · RRF (Cormack et al. 2009) · shared-norm multi-paragraph reading (Clark & Gardner
2018) · SQuAD 2.0 (Rajpurkar et al. 2018) · multi-task uncertainty weighting (Kendall et al. 2018) · selective
classification with guaranteed risk (Geifman & El-Yaniv 2017) · conformal prediction (Angelopoulos & Bates
2023) · calibration (Guo et al. 2017) · Okapi BM25 (Robertson & Zaragoza 2009) · SIF (Arora et al. 2017) ·
BanglaRQA (Ekram et al. 2022) · BanglaBERT (Bhattacharjee et al. 2022).

---

## Appendix A — v1 → v4 task map (so "do T3" still means what it meant)

| v1 task | v4 | Tier | Note |
|---|---|---|---|
| T0 skeleton | **T0** | A | + `reports/env.json` |
| T1 BanglaRQA ingest | **T1** | A | now raw-JSON; **+ alignment audit** |
| T2 NCTB ingest | **T2** | A | hash pinned; Mendeley risk closed |
| T2b 500M corpus | **X1** | **B** | downloaded shards, not streamed |
| T3 index assembly | **T3** | A | unchanged |
| T4 preprocessing + SPM tokenizer | **T4** (rules) + **X2** (SPM) | A / **B** | split: rule preprocessing is Tier A, SentencePiece is Tier B |
| T5 sparse retrieval | **T5** | A | unchanged |
| T6 Word2Vec/FastText | **X3** | **B** | |
| T6b synonym resources | **T6b** | A | hand-authored in Tier A; X3 adds induced candidates |
| T7 static-dense + hybrid | **T7** (hybrid) + **X4** (static-dense) | A / **B** | split |
| T8 transformer MLM | **X5** | **B** | + MorphSpan-MLM, pinned checkpoints |
| — | **X6** DAPT | **B** | new |
| T9 dual-encoder | **X7** | **B** | + X7b self-mined negatives |
| — | **X8** late interaction + cross-encoder | **B** | new |
| T10 span reader + ladder | **T10** (feature reader, tiers 1–3) + **X9** (neural reader) | A / **B** | the Tier-A reader is the neural reader's baseline |
| T11 verifier | **T11** + **T11b** BanglaVerify + **X10** neural S3 | A / **B** | + S5, conformal |
| T12 pipeline + demo | **T12** + **V1–V4** | A | verifiability split out |
| T13 full evaluation | **T13** + **T13b** + **T14** | A | T13b is RQ-A |
| T14 report | **T15** | A | same structure |
| E1 generative reader | **E1** | post | unchanged |
| E2 grading tab | — | — | **dropped in v4** (§12); the veto layer and contrast set it needed live in T11 |
| — | **X11 X12 X13** | **B** | objective study · scaling + probing · scratch-vs-pretrained |

## Appendix B — every hyperparameter in one place (mirrors `config.py`)

```python
SEED = 1337

# ── corpus ──────────────────────────────────────────────────────────────────
MIN_PASSAGE_CHARS   = 40
MIN_BENGALI_RATIO   = 0.60
DEDUP_JACCARD       = 0.85
INDEX_SIZES         = (10_000, 50_000)   # nested; all 3,000 gold contexts in both
WIKI_CHUNK_TOKENS   = (120, 180); WIKI_CHUNK_OVERLAP = 30
TOKEN_BUDGET        = 500_000_000        # [Tier B]
BCORE_SHARDS        = 2                  # [Tier B]
DOMAIN_UPWEIGHT     = 3                  # NCTB + BanglaRQA contexts

# ── Tier A: sparse retrieval ────────────────────────────────────────────────
BM25_K1 = 1.2; BM25_B = 0.75             # tuned on val
TFIDF_WORD_NGRAMS = (1, 2); TFIDF_CHAR_NGRAMS = (3, 5)
RRF_K = 60; TOP_K = 10

# ── Tier A: reader ──────────────────────────────────────────────────────────
CAND_SENTENCES = 3; CAND_NGRAM_MAX = 12; SPAN_MAX_TOKENS = 30
READER_MODELS = ("logreg", "hist_gbdt")
NULL_THRESHOLD = "tuned_on_val"

# ── Tier A: verification ────────────────────────────────────────────────────
VETO_CATEGORIES = ("numeral", "date", "unit", "entity", "polarity", "relation_order")
SIGNALS = ("s1_conf", "s2_ground", "s3_support", "s4_consistency", "s5_disagreement")
FUSION_MODELS = ("gaussian_nb", "logreg", "hist_gbdt")     # RQ5
CALIBRATORS   = ("none", "temperature", "platt", "isotonic")
CONFORMAL_ALPHA = 0.10      # target error rate among answered
CONFORMAL_DELTA = 0.10      # confidence in the bound
CONTRAST_SET_SIZE = 500
BANGLAVERIFY_VERIFY_SAMPLE = 300

# ── Tier A: evaluation ──────────────────────────────────────────────────────
BOOTSTRAP_RESAMPLES = 1_000
HEADLINE_INDEX = 50_000
HEADLINE_METRIC = "token_f1"          # tier 2
COUNTERFACTUAL_N = 30

# ── Tier B: tokenizer ───────────────────────────────────────────────────────
SPM_VOCAB = 32_000; SPM_MODEL = "unigram"
SPM_INPUT_SENTENCES = 10_000_000; CHAR_COVERAGE = 0.9995

# ── Tier B: static embeddings ───────────────────────────────────────────────
EMB_DIM = 300; WINDOW = 5; MIN_COUNT = 5; NEGATIVE = 10; SG = 1; EPOCHS = 3
FASTTEXT_NGRAMS = (3, 6); FASTTEXT_BUCKET = 1_000_000

# ── Tier B: encoder ─────────────────────────────────────────────────────────
N_LAYERS = 6; D_MODEL = 384; N_HEADS = 6; D_FF = 1536; MAX_LEN = 256
DROPOUT = 0.1; POSITIONS = "rope"; TIE_EMBEDDINGS = True; INIT_STD = 0.02

# ── Tier B: MLM pretraining ─────────────────────────────────────────────────
MASK_RATE = 0.15; REPLACE = (0.8, 0.1, 0.1)
MORPH_PROB = 0.5; SPAN_GEOMETRIC_P = 0.3; SPAN_MAX_WORDS = 4; SALIENCE_LAMBDA = 0.3
LR = 3e-4; WARMUP_STEPS = 2_000; LR_MIN = 3e-5; SCHEDULE = "cosine"
BATCH = 32; GRAD_ACCUM = 4; CLIP = 1.0; PRECISION = "bf16"
ADAMW = dict(betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
CKPT_EVERY_STEPS = 1_000; TIME_BOX_HOURS = 6
PINNED_TOKEN_CHECKPOINTS = (50e6, 100e6, 200e6, 350e6, 500e6)
DAPT_TOKENS = 50_000_000

# ── Tier B: objective study (equal compute) ─────────────────────────────────
OBJECTIVE_ARMS = ("mlm_random", "mlm_wwm", "morphspan", "electra_rtd")
STUDY_TOKENS   = 50_000_000
ELECTRA_GEN    = dict(n_layers=3, d_model=256, rtd_weight=50)

# ── Tier B: neural retrieval & reader ───────────────────────────────────────
INFONCE_TEMP = 0.05; HARD_NEGS = 2; DE_LR = 2e-5; DE_EPOCHS = 3
TOP_K_FUSE = 100; TOP_K_LATE = 20; TOP_K_CROSS = 20; COLBERT_DIM = 128
SHARED_NORMALISATION = True; MULTITASK = True; ENSEMBLE_SEEDS = (1337, 2024, 7)
```

## Appendix C — the first commands

```powershell
# 1. environment — fixes the cp1252 Bangla crash and keeps downloads off the system drive
$env:PYTHONIOENCODING = "utf-8"; chcp 65001
$env:HF_HOME = "E:\NLP\hf_cache"

# 2. TIER A dependencies — no torch, no transformers, ~40 MB. Tier A is buildable right now.
python -m pip install scikit-learn pandas numpy scipy pyarrow datasketch `
                      matplotlib seaborn tqdm streamlit pytest

# 3. TIER B dependencies — on the RTX 3080 machine only (cu121 or cu124 to match the driver)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers gensim sentencepiece bert-score

# 4. confirm the GPU is real before planning six hours around it
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 5. and then, one task at a time
#    "do T0"
```

---

**Tonight's jobs, in order: T0 → T1 → T2 → T3 → T4 → T6b (Tier A, CPU, ~6 h), and in parallel on the GPU
machine X1 → X2 → launch X5 before you sleep.** Tier A's demo does not depend on X5 finishing; X5 just needs to
start tonight to have a chance of landing by the 22nd.

Say **"do T0"** and I will build the skeleton, verify the environment into `reports/env.json`, and stop.

# TASKS — Tier A build log

Status of the tasks in `PLAN.md` §14. One task at a time; **done** means the
task's *Done when* criterion in PLAN.md §14 is met and the artefact exists.

Every number below is also in `reports/tables/` or `reports/results.json`,
written by an evaluation script and stamped with the config hash and seed that
produced it (VC-12). Nothing here is typed by hand from memory.

| Task | Status | Artefacts | Notes |
|---|---|---|---|
| **T0** Repo skeleton, config, seeds, env | ✅ done | `src/bnqa/`, `config.py`, `reports/env.json`, `requirements-core.txt` | Tier A installs with **no torch / transformers / gensim** |
| **T1** BanglaRQA ingest, leakage gate, alignment audit | ✅ done | `qa_{train,val,test}.jsonl`, `gold_contexts.jsonl`, `tables/splits.csv`, `tables/answer_alignment.csv` | counts match 11,912 / 1,484 / 1,493; leakage gate **CLEAN**; span alignment **97.9%** |
| **T2** NCTB textbook ingest | ✅ done | `nctb_passages.jsonl`, `tables/nctb_ingest.csv`, `data/sources.csv` | 43,359 SchoolText passages, grades 6–10, 32 subjects |
| **T3** Index assembly + dedup + audit | ✅ done | `passages.jsonl`, `index_{10,50}k.manifest.json`, `tables/corpus_audit.csv` | nested, all 3,000 gold at both sizes, 20,000 citable NCTB passages |
| **T4** Normalisation, tokeniser, stopwords, stemmer | ✅ done | `bnqa/preprocess/` | idempotent; stemmer is corpus-validated |
| **T5** Sparse retrieval (TF-IDF word/char, our BM25) | ✅ done | `bnqa/retrieval/`, `tables/retrieval.csv` | BM25 validated against a hand-worked example in `pytest` |
| **T6b** Lexical resources + probe | ✅ done | `resources/{suffixes,synonyms,variants,synonym_probe}`, `tables/synonym_probe.csv` | precision **1.000** (0 false merges on 29 hard negatives), recall **0.714** |
| **T7** RRF hybrid, query expansion, index-size comparison, topics | ✅ done | `tables/retrieval.csv`, `tables/retrieval_index_size.csv`, `tables/retrieval_length_control.csv`, `figures/topics_50k.png` | RQ1 answered on val at both sizes |
| **T10** Reader: question type, candidates, feature span ranker | ✅ done | `tables/reader.csv`, `tables/reader_features.csv`, `tables/qtype.csv` | tier-2 F1 **0.227** (gbdt), question-type head **0.902**; candidate recall **0.472** is the binding ceiling |
| **T11** Verifier: veto, S1–S5, fusion, calibration, conformal | 🟡 code done, **evaluation must re-run** | `tables/{fusion,calibration}.csv` written; `conformal`/`abstention_policies` **not yet** | fusion (RQ5) and the calibration study completed; the run then died on a pickling bug before conformal. Both are now fixed — see findings 15–16 |
| **T11b** BanglaVerify + S3 | 🟡 code done, **must re-run** | `banglaverify_*.jsonl` regenerated; `tables/banglaverify_quality.csv` | the first build was invalid (finding #15); construction fixed and spot-checked, full build pending |
| **T12** `pipeline.py` + 3-tab Streamlit | ✅ done | `src/bnqa/pipeline.py`, `app/streamlit_app.py` | the six-tuple, all three tabs, the VC-7 sidebar toggle; end-to-end ask + receipt verified |
| **V1** Evidence receipts + `verify_receipt.py` | 🟡 code done, **not yet run at scale** | `scripts/verify_receipt.py`, `tests/test_span_is_substring.py` | the receipt round-trip and its tamper check pass in `pytest`; `reports/receipts/` is filled by T13 |
| **V2** Offline mode + determinism | ✅ done | `src/bnqa/offline.py`, `tests/test_determinism_and_offline.py` | `BNQA_OFFLINE=1` raises on any outbound connection |
| **V3** Counterfactual corpus test | 🟡 code done, **not yet run** | `scripts/run_counterfactual.py`, `bnqa/eval/counterfactual.py` | needs the fitted verifier, so it follows the T11 re-run |
| **V4** Teacher audit sheet + model card | 🟡 code done, **not yet run** | `scripts/make_report_assets.py` | same dependency |
| **T13** Evaluation pass → `results.json` | 🟡 partial | `reports/results.json`, `scripts/run_all.ps1` | T0–T7 and T10 are current and single-hash; T11 onward pending |
| ~~G1–G3 Grading~~ | ❌ removed | | out of scope — PLAN.md §12 |

**Where Tier A actually stands.** Every Tier-A component is written, committed
and unit-tested (91 tests). T0–T7 and T10 have been run end to end at the 50k
headline index and their numbers are in `reports/`. **T11 onward needs one more
run**: the first attempt produced an invalid S3 dataset (finding #15) and then
crashed on a pickling bug (finding #16). Both defects are fixed and verified on
a reduced build; the full pass is `scripts/run_all.ps1` from T11 down, about an
hour. Tier B (X1–X13) has not been started.

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

3. **`class` is the string `"9-10"`** for the combined Nine–Ten book — 23,004
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

8. **The index had a length artefact, it was ours, and only half of it is
   fixable.** Chunking Wikipedia to a fixed 120–180 tokens while leaving
   BanglaRQA contexts whole once made the two trivially separable: **passage
   length alone predicted "is gold" at AUC 0.795**. Sampling each distractor's
   target length from the gold distribution fixed that, and it stays fixed —
   the audit now measures it on every build and Wikipedia sits at **AUC 0.546**,
   essentially chance.

   But the *overall* figure is **0.726**, because NCTB-SchoolText separates at
   **AUC 0.968**: its passages are natively much shorter (median 57 tokens
   against gold's 203) and T3 deliberately keeps its own chunking, because that
   chunking is what carries the grade/subject/chapter metadata VC-2 cites.
   There is no fix that does not destroy the citation unit, so the number is
   reported instead of removed, per source rather than pooled.

   **It is not cosmetic, and we now know what it is worth.** BM25's `b` tunes
   length normalisation directly, and it tunes toward weak normalisation — which
   favours long documents, which are the gold ones — at *every* `k1`. Mean
   Recall@5 at `b=0.3` versus `b=0.9` is **+0.0088 at 10k** and **+0.0187 at
   50k**, the gap growing with NCTB's share of the index. `run_retrieval.py
   --control-only` re-runs BM25 with NCTB removed so the headline can be read
   beside a task whose distractors are known to be distributionally matched.

9. **The hand-authored lexicon barely fires.** Query expansion changes Recall@5
   by less than a point, because the T6b lexicon is curriculum-domain and
   BanglaRQA's questions are Wikipedia-domain. This is the honest Tier-A
   baseline that Tier B's induced synonyms (X3) have to beat — it makes RQ-A
   measurable rather than rhetorical.

10. **The synonym probe was circular, and the fix needs stating carefully.**
    Every positive had been copied from the lexicon, so recall was 1.0 by
    construction. With held-out Bangla doublets added, the honest numbers are
    **precision 1.000 (0 false merges across 29 hard negatives — antonyms and
    co-hyponyms the veto layer must also keep apart) and recall 0.714**.

    The pairs whose terms the lexicon has *never seen* score 0.000 recall, and
    reporting that as a result would be theatre: equivalence is a union-find
    lookup, so those pairs cannot match **by definition**. That row is labelled
    as what it actually measures — the coverage a 56-pair hand list does not
    reach, which is exactly the gap X3 must close on the same probe.

11. **The veto layer missed the plan's own flagship contradiction.** PLAN.md
    §10.1 row 1 is *উদ্ভিদ **অক্সিজেন** গ্রহণ করে* against *উদ্ভিদ **কার্বন
    ডাই-অক্সাইড** গ্রহণ করে*. Neither substance is a proper noun, so the
    rule-based entity proxy never reached it, and every soft metric scores the
    pair at ≈1.0 — the exact failure the veto layer exists to prevent.

    Fixed with an **exact minimal-pair term-swap test**: set arithmetic over
    content tokens, firing only when the two strings are otherwise near-copies.
    No edit distance and no n-gram overlap, so the isolation guarantee in
    §10.2 holds and `tests/test_veto_isolation.py` asserts it. It consults the
    T6b synonym classes first, so a genuine paraphrase (বায়ুমণ্ডল / আবহমণ্ডল) is
    **not** vetoed. BanglaVerify's entity generator was extended the same way,
    because a generator restricted to proper nouns could never produce the pair
    the detector most needs to be trained against.

12. **Two follow-on veto bugs, both found by asking real questions.** The
    gazetteer is harvested from passage titles, so four-digit years leak into
    it and a swapped year was reported as *নাম ভুল* — "wrong name" — for a
    date. Numbers are now excluded from the entity proxy. With that fixed, the
    date rule and the term-swap rule both fired on the same year and printed
    the contradiction twice; term swap now ignores numeric tokens, which are
    not its job.

13. **Candidate recall is the reader's binding constraint.**
    Measured over train questions, the gold span was absent from the candidate
    set for nearly half of them — a hard ceiling on the reader that no amount
    of ranking could lift. Breakdown: gold sentence outside the top-3 (21%),
    **span starts or ends on a stopword (12%)**, **span longer than 12 tokens
    (9%)**, type filter (3%).

    The middle two were wrong. Rejecting stopword-edged spans looked sensible
    and simply deleted correct answers, because Bangla answers routinely start
    with a determiner or end with a case particle; the ranker already carries
    `starts_with_stopword` and `stopword_frac`, and a feature it can weigh
    beats a filter it cannot appeal. And REASON/PROCESS answers are *clauses*
    (§7.5) that no 12-token n-gram can reach, so clause-shaped candidates are
    now generated whole for those classes.

    The top-3-sentence limit is PLAN.md's own and was left alone. After the
    fixes, candidate recall measured over the real retrieval output at 50k is
    **0.472** — 1,585 of 3,000 training questions have no gold span among
    their candidates and can teach the ranker nothing. That is a hard ceiling,
    and it is why tier-2 token F1 is only **0.227** (GBDT; logreg 0.223,
    strict EM 0.114). The reader is the weakest link in Tier A, the reason is
    measured rather than guessed, and it is now a reported column in
    `reader.csv` instead of an invisible ceiling.

    The feature table behaves sensibly, which is the point of having one:
    `cos_q_sentence` (+1.19) and `passage_rank_inv` (+0.99) are the strongest
    positives, while `cos_q_span` (−0.69) and `span_in_question` (−0.37) are
    negative — a span that merely echoes the question is usually wrong.

14. **The retrieval hybrid does not pay off, and query expansion does nothing.**
    On val at the 50k headline index: BM25 **0.891** Recall@5 at 0.53 ms/query
    and 43.6 MB; `rrf(bm25+char)` **0.873** at 248.51 ms and 520 MB;
    `tfidf-char` **0.801** at 244.51 ms and 476 MB. BM25 wins on quality *and*
    costs two orders of magnitude less, so it is the shipped arm.

    RRF *hurts* because rank fusion assumes its arms are comparable: folding a
    0.801 arm into a 0.891 arm drags ranks down. Query expansion moves Recall@5
    by 0.001, confirming finding #9. Index size behaves as §6.2B predicts —
    every arm degrades from 10k to 50k, char TF-IDF worst (−10.0% relative),
    BM25 most robust (−2.7%).

    The NCTB control (finding #8) puts a bound on the remaining artefact:
    dropping all 20,000 NCTB passages — 40% of the index — moves Recall@5 by
    only **+0.0135**, so those passages are weak distractors but not free ones.

15. **S3 scored 1.0000 on a contrast set built to be hard, and that was a bug
    report rather than a result.** BanglaVerify paired each SUPPORTED claim
    with the very sentence it was copied from, so **6,950 of 20,429 training
    pairs were byte-identical strings**. The classifier learned
    `statement == evidence` and got perfect accuracy for free — on a task the
    pipeline never presents it with, because there the claim is an extracted
    span and the evidence is a retrieved passage.

    Claims are now judged against their **containing passage**. Identical pairs
    fall to **0**, and the honest numbers on a reduced build are val **0.938**
    (majority 0.409) and **contrast 0.728 against a 0.672 majority** — barely
    above baseline, with CONTRADICTED recall only 0.610. That is exactly the
    outcome §10.7 tells us to report rather than bury: on held-out minimal
    pairs the lexical classifier is close to a similarity detector, **which is
    the argument for the veto layer**, not against it.

    The change also exposed that generator quality must be scored at sentence
    level: checking a negated claim against a whole passage made the polarity
    generator look like 0.750, because almost any passage contains a negation
    somewhere. Scored against the sentence it was perturbed from — the
    comparison the veto layer actually performs — it is **0.995**. The entity
    generator sits at **0.872**, below PLAN.md §10.3's ~90% floor, and is
    flagged rather than quietly kept.

16. **The calibration study ran, then threw its own results away.** The fitted
    calibrator held a lambda closure, so `pickle.dump` raised
    `Can't get local object 'fit_calibrator.<locals>.<lambda>'` *after* all
    four methods had been compared — taking the conformal threshold, the
    policy comparison and the test pass down with it. Calibrators are now
    plain picklable objects. The comparison itself was sound: Platt
    **ECE 0.079**, isotonic 0.080, temperature 0.282, uncalibrated 0.291.

    Fusion (RQ5) completed and is worth keeping: logreg **AUROC 0.729**, GBDT
    0.722, **Naive Bayes 0.505 — chance**. The generative arm's independence
    assumption is badly violated here, which is the measured answer to Sir's
    Q5 rather than a paragraph about it.

17. **`results.json` had accumulated three different config hashes**, some
    written on a different machine, and a stale row is indistinguishable from a
    current one by inspection — which is precisely what VC-12 exists to
    prevent. `stale_results()` names them, `prune_results()` removes them, and
    the last step of `run_all.ps1` reports what it dropped.

## Commands

```powershell
. .\scripts\env.ps1                        # UTF-8 console + PYTHONPATH + HF_HOME
.\scripts\run_all.ps1                      # everything, from a clean checkout
.\scripts\run_all.ps1 -Quick               # ten-minute wiring check

python -m bnqa.envinfo                     # T0   -> reports/env.json
python -m bnqa.data.fetch                  # download the Tier A corpora (~550 MB)
python -m bnqa.data.banglarqa              # T1
python -m bnqa.data.nctb_text              # T2
python -m bnqa.data.wiki                   # T3a  length-matched distractors
python -m bnqa.data.index_build            # T3   nested indexes
python -m bnqa.data.audit                  # T3   corpus audit + length artefact
python -m bnqa.eval.lexicon_probe          # T6b
python scripts/run_retrieval.py            # T5 + T7
python scripts/run_retrieval.py --control-only   # T7 length-artefact control
python -m bnqa.eval.topics                 # T7   topic figure
python scripts/run_reader.py               # T10
python scripts/run_verify.py               # T11 + T11b
python scripts/run_counterfactual.py       # V3
python scripts/make_report_assets.py       # V1 + V4 + T13
python scripts/verify_receipt.py           # VC-3, on any machine
pytest                                     # all 91 invariants
streamlit run app/streamlit_app.py         # T12
```

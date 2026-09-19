"""পাঠ-প্রমাণ — the demo UI (PLAN.md §7.9, task T12).

    streamlit run app/streamlit_app.py

Three tabs, Bangla-first, plain and fast:

* **প্রশ্ন করুন** — ask; answer in large type, confidence bar with the conformal
  τ marked, the supporting sentence highlighted inside its passage, the
  citation line, the five signal bars, the top passages, an explicit
  abstention message **that still shows the closest passage**, the veto reason
  when one fired, the প্রমাণ দেখুন (receipt) button, and the উত্তর যাচাই control
  that checks a candidate answer a student types.
* **কর্পাস অনুসন্ধান** — search the corpus, filter by grade/subject/chapter, and
  ask a question against a passage **the examiner chose**.  This tab exists
  purely so a teacher can audit us.
* **সিস্টেম তথ্য** — the numbers: corpus hashes, config hash, seed, the
  conformal guarantee and its achieved risk, the counterfactual results.  The
  "we are not a GPT wrapper" tab.

The sidebar carries VC-7's **কর্পাস: আসল / পরিবর্তিত** toggle.  Flipping it
rebuilds the pipeline against the tampered corpus, so the answer visibly
follows the book rather than the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bnqa.config import (CFG, COUNTERFACTUAL, ENV_JSON, PASSAGES,  # noqa: E402
                         RESULTS_JSON, SOURCES_CSV, config_hash)
from bnqa.offline import status as offline_status  # noqa: E402
from bnqa.pipeline import BanglaQA  # noqa: E402
from bnqa.preprocess.normalize import normalize, to_bengali_digits  # noqa: E402
from bnqa.utils import load_jsonl, read_json  # noqa: E402

st.set_page_config(page_title="পাঠ-প্রমাণ · Path-Proman", page_icon="📖", layout="wide")

TAMPERED = COUNTERFACTUAL / "passages_tampered.jsonl"


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="কর্পাস ও মডেল লোড হচ্ছে…")
def load_pipeline(size: int, retriever: str, tampered: bool) -> BanglaQA:
    from bnqa.retrieval.index import prepare

    corpus = list(prepare(size, verbose=False))
    if tampered and TAMPERED.exists():
        swaps = {p["pid"]: p for p in load_jsonl(TAMPERED)}
        corpus = [swaps.get(p["pid"], p) for p in corpus]
    return BanglaQA.load(size=size, corpus=corpus, retriever=retriever)


@st.cache_data(show_spinner=False)
def load_results() -> dict:
    return read_json(RESULTS_JSON) if RESULTS_JSON.exists() else {}


@st.cache_data(show_spinner=False)
def corpus_facets(size: int) -> dict:
    from bnqa.retrieval.index import prepare

    passages = prepare(size, verbose=False)
    grades = sorted({str(p.get("grade_label") or p.get("grade")) for p in passages
                     if p.get("grade_label") or p.get("grade")})
    subjects = sorted({p.get("subject") for p in passages if p.get("subject")})
    return {"grades": grades, "subjects": subjects, "n": len(passages)}


# --------------------------------------------------------------------------- #
# Sidebar                                                                      #
# --------------------------------------------------------------------------- #

st.sidebar.title("📖 পাঠ-প্রমাণ")
st.sidebar.caption("উত্তর নয় — প্রমাণসহ উত্তর.")

size = st.sidebar.selectbox("সূচক (index)", sorted(CFG.index_sizes),
                            index=len(CFG.index_sizes) - 1,
                            format_func=lambda n: f"{n // 1000}k passages")
retriever_kind = st.sidebar.selectbox(
    "রিট্রিভার", ["bm25", "rrf", "tfidf-word", "tfidf-char"],
    help="RQ1 live: switch the retrieval arm and watch the evidence change.")

tampered = st.sidebar.radio(
    "কর্পাস", ["আসল (original)", "পরিবর্তিত (tampered)"],
    help="VC-7 — the counterfactual corpus test.  Flip it and the answer should "
         "follow the book, not the model.",
    disabled=not TAMPERED.exists()) == "পরিবর্তিত (tampered)"
if tampered:
    st.sidebar.warning("পরিবর্তিত কর্পাস সক্রিয় — উত্তর বই অনুসরণ করবে।")
elif not TAMPERED.exists():
    st.sidebar.caption("পরিবর্তিত কর্পাস নেই — `scripts/run_counterfactual.py` চালান।")

off = offline_status()
if off["guard_installed"]:
    st.sidebar.success("BNQA_OFFLINE=1 — নেটওয়ার্ক বন্ধ (VC-6)")

qa = load_pipeline(size, retriever_kind, tampered)
results = load_results()

tau = qa.conformal.tau if (qa.conformal and qa.conformal.feasible) else 0.5

tab_ask, tab_corpus, tab_system = st.tabs(
    ["প্রশ্ন করুন", "কর্পাস অনুসন্ধান", "সিস্টেম তথ্য"])


# --------------------------------------------------------------------------- #
# Shared rendering                                                             #
# --------------------------------------------------------------------------- #


def highlight(passage_text: str, start: int, end: int, s_start: int, s_end: int) -> str:
    """The supporting sentence, with the answer span marked inside it."""
    text = normalize(passage_text)
    sent = text[s_start:s_end]
    rel_s, rel_e = start - s_start, end - s_start
    if 0 <= rel_s <= rel_e <= len(sent):
        sent = (sent[:rel_s] + "**:orange[" + sent[rel_s:rel_e] + "]**" + sent[rel_e:])
    return sent


def render_result(res, *, key: str) -> None:
    if res.abstained:
        st.error(f"### 🚫 {res.message or 'উত্তর দেওয়া হচ্ছে না'}")
        st.caption("যে অনুচ্ছেদটি সবচেয়ে কাছাকাছি ছিল সেটি নিচে দেখানো হলো — "
                   "অস্বীকার মানে লুকানো নয়।")
    else:
        st.markdown(f"# {res.answer}")
        if res.yes_no:
            st.markdown(f"### উত্তর: **{res.yes_no}**")

    # ---- confidence with τ marked ---------------------------------------
    c1, c2 = st.columns([3, 1])
    with c1:
        st.progress(min(max(res.confidence, 0.0), 1.0))
        st.caption(f"আত্মবিশ্বাস {res.confidence:.3f}  ·  সীমা τ = {tau:.3f}  "
                   f"{'✅ উপরে' if res.confidence >= tau else '⚠️ নিচে'}")
    with c2:
        st.metric("সময়", f"{res.latency_ms:.0f} ms")

    if res.veto_reason:
        st.warning(f"⛔ ভেটো: {res.veto_reason}")

    # ---- evidence --------------------------------------------------------
    if res.pid:
        passage = qa.by_pid.get(res.pid)
        if passage:
            st.markdown("#### সমর্থনকারী বাক্য")
            st.info(highlight(passage["text"], res.char_start, res.char_end,
                              res.sentence_start, res.sentence_end)
                    if res.answer else res.evidence)
            st.markdown(f"**উদ্ধৃতি:** `{res.citation_line()}`")

            ok = normalize(passage["text"])[res.char_start:res.char_end] == res.answer
            if res.answer:
                st.caption(("✅ VC-1: `passages[pid].text[start:end] == answer`"
                            if ok else "❌ VC-1 ব্যর্থ — এটি রিপোর্ট করুন"))

            with st.expander("সম্পূর্ণ অনুচ্ছেদ দেখুন"):
                st.write(normalize(passage["text"]))

    # ---- the five signal bars -------------------------------------------
    st.markdown("#### পাঁচটি সংকেত (signals)")
    cols = st.columns(5)
    labels = {"s1_reader": "S1 রিডার", "s2_grounding": "S2 ভিত্তি",
              "s3_support": "S3 সমর্থন", "s4_consistency": "S4 সঙ্গতি",
              "s5_ensemble": "S5 ঐকমত্য"}
    for col, (name, label) in zip(cols, labels.items()):
        value = float(res.signals.get(name, 0.0))
        col.metric(label, f"{value:.2f}")
        col.progress(min(max(value, 0.0), 1.0))

    # ---- top passages ----------------------------------------------------
    with st.expander(f"শীর্ষ {len(res.passages)} অনুচ্ছেদ"):
        for p in res.passages:
            mark = "👉 " if p["pid"] == res.pid else ""
            st.markdown(f"{mark}**{p['rank']}. `{p['pid']}`**  "
                        f"score {p['retriever_score']:.4f}  ·  {p.get('source')}")
            st.caption(normalize(p["text"])[:300] + "…")

    # ---- the receipt -----------------------------------------------------
    if st.button("🔍 প্রমাণ দেখুন (Verify)", key=f"receipt_{key}"):
        from bnqa.verify.receipt import check_receipt

        report = check_receipt(res.receipt, {p["pid"]: normalize(p["text"])
                                             for p in qa.corpus})
        for c in report["checks"]:
            (st.success if c["pass"] else st.error)(f"{c['check']}: {c['detail']}")
        st.json(res.receipt)


# --------------------------------------------------------------------------- #
# Tab 1 — Ask                                                                  #
# --------------------------------------------------------------------------- #

with tab_ask:
    st.subheader("প্রশ্ন করুন")
    question = st.text_input("আপনার প্রশ্ন (বাংলায়)",
                             placeholder="যেমন: সালোকসংশ্লেষণে উদ্ভিদ কোন গ্যাস গ্রহণ করে?")
    if st.button("উত্তর খুঁজুন", type="primary") and question.strip():
        with st.spinner("খোঁজা হচ্ছে…"):
            st.session_state["last"] = qa.ask(question)
    if st.session_state.get("last") is not None:
        render_result(st.session_state["last"], key="ask")

    st.divider()
    st.subheader("উত্তর যাচাই — একটি উত্তর বই মিলিয়ে দেখুন")
    st.caption("একটি সম্ভাব্য উত্তর লিখুন; কর্পাস সেটি সমর্থন করে কি না, এবং না করলে "
               "কোন শব্দটি বইয়ের সঙ্গে মেলে না, তা জানানো হবে।")
    vc1, vc2 = st.columns(2)
    v_question = vc1.text_input("প্রশ্ন", key="verify_q")
    v_answer = vc2.text_input("যে উত্তরটি যাচাই করবেন", key="verify_a")
    if st.button("যাচাই করুন") and v_question.strip() and v_answer.strip():
        out = qa.verify_claim(v_question, v_answer)
        (st.success if out["supported"] else st.error)(
            f"{'✅ কর্পাস সমর্থন করে' if out['supported'] else '❌ সমর্থিত নয়'} — {out['reason']}")
        st.markdown(f"**বইয়ের বাক্য:** {out['evidence']}")
        st.caption(f"`{out['pid']}`  ·  আত্মবিশ্বাস {out.get('confidence', 0):.3f}"
                   + (f"  ·  S3: {out['label']}" if out.get("label") else ""))
        if out["veto"].get("fired"):
            for d in out["veto"]["detail"]:
                st.warning(f"⛔ {d['reason']}")


# --------------------------------------------------------------------------- #
# Tab 2 — Corpus explorer                                                      #
# --------------------------------------------------------------------------- #

with tab_corpus:
    st.subheader("কর্পাস অনুসন্ধান")
    st.caption("এই ট্যাবটি আছে যাতে একজন শিক্ষক নিজে যাচাই করতে পারেন — "
               "আপনি অনুচ্ছেদ বেছে নিন, আমরা সেটির বিরুদ্ধেই উত্তর দেব।")

    facets = corpus_facets(size)
    f1, f2, f3 = st.columns(3)
    grade = f1.selectbox("শ্রেণি", ["সব"] + facets["grades"])
    subject = f2.selectbox("বিষয়", ["সব"] + facets["subjects"])
    query = f3.text_input("খুঁজুন (free text)")

    hits = []
    for p in qa.corpus:
        if grade != "সব" and str(p.get("grade_label") or p.get("grade")) != grade:
            continue
        if subject != "সব" and p.get("subject") != subject:
            continue
        if query and query not in p["text"]:
            continue
        hits.append(p)
        if len(hits) >= 50:
            break

    st.caption(f"{len(hits)} টি অনুচ্ছেদ দেখানো হচ্ছে (সর্বোচ্চ ৫০)  ·  "
               f"সূচকে মোট {facets['n']:,d}")
    for p in hits[:20]:
        with st.expander(f"`{p['pid']}` · {p.get('subject') or p.get('source')} · "
                         f"{p.get('chapter_title') or ''}"):
            st.write(normalize(p["text"]))
            q = st.text_input("এই অনুচ্ছেদ থেকে প্রশ্ন করুন", key=f"q_{p['pid']}")
            if st.button("উত্তর", key=f"b_{p['pid']}") and q.strip():
                single = BanglaQA(corpus=[p], retriever=qa.retriever, reader=qa.reader,
                                  support_model=qa.support_model,
                                  fusion_model=qa.fusion_model,
                                  calibrator=qa.calibrator, conformal=qa.conformal,
                                  policy=qa.policy, ensemble=qa.ensemble)
                # The retriever indexes the whole corpus; restricting the
                # *reader* to the chosen passage is what "ask against this
                # passage" means, so the answer cannot come from anywhere else.
                res = single.reader.read(q, [{**p, "rank": 1, "retriever_score": 1.0}],
                                         ranks=[(1, 1.0)])
                st.markdown(f"### {res.text or '—'}")
                st.caption(f"chars {res.char_start}–{res.char_end}  ·  "
                           f"score {res.score:.3f}")
                st.info(res.sentence)


# --------------------------------------------------------------------------- #
# Tab 3 — System card                                                          #
# --------------------------------------------------------------------------- #

with tab_system:
    st.subheader("সিস্টেম তথ্য — System card")
    st.caption("We are not a GPT wrapper.  Every number here is written by an "
               "evaluation script, never typed by hand (VC-12).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Config hash", config_hash())
    c2.metric("Seed", CFG.seed)
    c3.metric("Index", f"{size // 1000}k")
    c4.metric("Retriever", retriever_kind)

    conf = results.get("t11_verify", {}).get("summary", {}).get("conformal")
    if conf:
        st.markdown("#### ⚖️ The guarantee (VC-14)")
        g1, g2, g3 = st.columns(3)
        g1.metric("τ", f"{conf.get('tau', float('nan')):.3f}")
        g2.metric("Achieved risk (test)", f"{conf.get('achieved_selective_risk', 0):.3f}")
        g3.metric("Coverage (test)", f"{conf.get('coverage', 0):.3f}")
        st.info(conf.get("claim", ""))
        st.caption("bound " + ("HELD ✅" if conf.get("bound_held") else "VIOLATED ❌"))

    cf = results.get("v3_counterfactual", {}).get("summary")
    if cf:
        st.markdown("#### 🏆 Counterfactual corpus test (VC-7)")
        k1, k2, k3 = st.columns(3)
        k1.metric("CAR", f"{cf.get('car', 0):.3f}",
                  help="Corpus Attribution Rate — the answer follows the corpus")
        k2.metric("AoRR", f"{cf.get('aorr', 0):.3f}",
                  help="Abstention on Removal — deleting the passage makes it abstain")
        k3.metric("N", cf.get("n", 0))

    st.markdown("#### Corpus")
    if SOURCES_CSV.exists():
        import pandas as pd

        st.dataframe(pd.read_csv(SOURCES_CSV)[["name", "license", "bytes", "sha256"]],
                     use_container_width=True, hide_index=True)

    st.markdown("#### Environment")
    if ENV_JSON.exists():
        st.json(read_json(ENV_JSON))

    st.markdown("#### Everything measured so far")
    st.json(results, expanded=False)

    st.caption(f"passages.jsonl → `{PASSAGES}`  ·  "
               f"সব সংখ্যা `{RESULTS_JSON.name}` থেকে")

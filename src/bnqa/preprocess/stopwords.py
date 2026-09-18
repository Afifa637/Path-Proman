"""Bangla stopwords — **sparse retrieval only** (PLAN.md §7.2, task T4).

Hand-authored: no downloaded list, so the Tier-A "nothing downloaded" claim in
PLAN.md §0.2 holds all the way down.

Two safety rules, both enforced by ``VETO_SAFE`` and a unit test:

* **Negation is never a stopword.**  ``না``/``নয়``/``নেই`` carry the polarity the
  veto layer checks (§10.2).  Removing them would make a sentence and its
  negation identical to the retriever — the exact failure the verifier exists to
  catch.
* **Numerals are never stopwords**, for the same reason.

Question words *are* stopwords here.  They are noise for retrieval, and the
reader's question analysis (§7.5) reads the raw question, never this list.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Words that must never be removed, whatever else changes                      #
# --------------------------------------------------------------------------- #

VETO_SAFE: frozenset[str] = frozenset({
    "না", "নয়", "নেই", "নাই", "ছাড়া", "বিনা", "ব্যতীত", "নি",
})

# --------------------------------------------------------------------------- #
# The list                                                                     #
# --------------------------------------------------------------------------- #

_STOPWORDS = """
এবং ও আর অথবা কিংবা বা তবে কিন্তু যদিও তথা এছাড়া অর্থাৎ তথাপি বরং
এই ঐ সেই এসব ওসব যেসব এটি এটা ওটা সেটি সেটা এগুলো সেগুলো ইহা তাহা
আমি আমরা আমার আমাদের তুমি তোমরা তোমার তোমাদের আপনি আপনারা আপনার
সে তারা তার তাদের তিনি তাঁরা তাঁর তাঁদের নিজে নিজের
যে যা যার যাদের যিনি যেটি যেটা যেখানে যখন যেহেতু যেন
হয় হল হলো হয়ে হয়েছে হয়েছিল হবে হত হতো হচ্ছে হইয়া হইতে
ছিল ছিলেন থাকে থাকা থেকে থাকেন রয়েছে আছে আছেন ছিলো
করা করে করেন করছে করেছে করলে করার করতে করবে কর
দেওয়া দিয়ে দিতে দেন দেয় নেওয়া নিয়ে নিতে
জন্য সঙ্গে সাথে দ্বারা মাধ্যমে পরে আগে পূর্বে মধ্যে ভিতরে ভেতরে বাইরে
উপর উপরে নিচে নীচে কাছে পাশে চারপাশে প্রতি বিরুদ্ধে অনুযায়ী মতো মত
একটি একটা একজন কিছু কিছুটা অনেক অনেকে সব সকল সমস্ত প্রত্যেক প্রতিটি
আরও আরো আবার এখন তখন কখনো সবসময় প্রায় খুব বেশ বেশি কম মাত্র শুধু কেবল
ইত্যাদি প্রভৃতি যেমন উদাহরণস্বরূপ অর্থে বিশেষত মূলত সাধারণত
তাই সুতরাং অতএব ফলে কারণে যদি তাহলে নতুবা নইলে
কি কী কে কারা কোন কোনো কোনটি কখন কোথায় কেন কীভাবে কিভাবে কত কয়টি কতটি
এর ওর তো ই ও হয়তো নাকি যেন বটে
""".split()

STOPWORDS: frozenset[str] = frozenset(_STOPWORDS) - VETO_SAFE


def is_stopword(token: str) -> bool:
    return token in STOPWORDS


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop stopwords.  Never drops a numeral or a negation."""
    return [t for t in tokens if t not in STOPWORDS]


def content_tokens(tokens: list[str]) -> list[str]:
    """Stopword-free, digit-bearing tokens kept.  Used for query expansion (T7)."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

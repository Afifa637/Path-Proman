"""Unsupervised topic structure over passages (PLAN.md §7.4, task T7).

The surviving unsupervised component, and the concrete answer to Sir's Q3 —
"is there room for unsupervised exploration?"  K-Means over TF-IDF passage
vectors with a silhouette sweep to choose k, plus a 2-D projection coloured by
cluster and a cross-tab against the known ``source`` labels.

The cross-tab is the part worth reading: if clusters align with *source* rather
than with *topic*, the index has a distribution artefact of exactly the kind
PLAN.md §6.2B warns about — a retriever could then partly succeed by learning
"gold passages look like Wikipedia".  So this figure doubles as a check on the
index composition.

    python -m bnqa.eval.topics
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..config import CFG, FIGURES
from ..retrieval.base import analyze_word
from ..utils import log_result, save_table, set_seed


def run(size: int | None = None, *, sample: int | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score

    set_seed()
    size = size or CFG.headline_index
    sample = sample or CFG.kmeans_sample

    from ..retrieval.index import prepare

    passages = prepare(size)
    rng = np.random.default_rng(CFG.seed)
    idx = rng.choice(len(passages), size=min(sample, len(passages)), replace=False)
    subset = [passages[i] for i in idx]
    print(f"T7c Topic clustering over {len(subset):,d} sampled passages")

    vec = TfidfVectorizer(analyzer=analyze_word, min_df=3, max_features=60_000,
                          sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(p["text"] for p in subset)
    svd = TruncatedSVD(n_components=100, random_state=CFG.seed)
    Z = svd.fit_transform(X)
    print(f"  TF-IDF {X.shape} -> SVD {Z.shape} "
          f"(explains {svd.explained_variance_ratio_.sum():.1%} of variance)")

    rows: list[dict] = []
    best = None
    for k in CFG.kmeans_clusters:
        km = KMeans(n_clusters=k, random_state=CFG.seed, n_init=10).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=min(4_000, len(Z)),
                               random_state=CFG.seed)
        rows.append({"k": k, "silhouette": round(float(sil), 4),
                     "inertia": round(float(km.inertia_), 2)})
        print(f"  k={k:3d}  silhouette {sil:.4f}")
        if best is None or sil > best[1]:
            best = (k, sil, km)

    assert best is not None
    k_best, sil_best, km_best = best
    print(f"  -> best k={k_best} (silhouette {sil_best:.4f})")

    # cluster x source cross-tab
    sources = sorted({p["source"] for p in subset})
    cross = np.zeros((k_best, len(sources)), dtype=int)
    for lab, p in zip(km_best.labels_, subset):
        cross[lab, sources.index(p["source"])] += 1
    purity = float((cross.max(axis=1).sum()) / cross.sum())

    # Purity over all three sources is the wrong alarm: NCTB is a genuinely
    # different domain from Wikipedia and is *supposed* to be separable — that is
    # the educational identity of the index, not a leak.  The artefact PLAN.md
    # §6.2B warns about is narrower: can anything tell the GOLD passages apart
    # from their same-distribution Wikipedia distractors?  If so, a retriever
    # could partly succeed by learning "gold passages look like this" and
    # Recall@k would be inflated without any relevance being modelled.
    #
    # Adjusted mutual information answers exactly that, and unlike purity it is
    # not fooled by the 1:9 class imbalance between gold and distractors.
    from sklearn.metrics import adjusted_mutual_info_score

    src_labels = [p["source"] for p in subset]
    ami_source = float(adjusted_mutual_info_score(src_labels, km_best.labels_))
    mask = [i for i, s in enumerate(src_labels) if s in ("gold", "wiki")]
    if mask:
        ami_gold_vs_wiki = float(adjusted_mutual_info_score(
            [src_labels[i] for i in mask], [km_best.labels_[i] for i in mask]))
    else:
        ami_gold_vs_wiki = float("nan")

    # top terms per cluster, for the report
    terms = np.array(vec.get_feature_names_out())
    centroids = svd.inverse_transform(km_best.cluster_centers_)
    top_terms = [", ".join(terms[np.argsort(-centroids[c])[:8]]) for c in range(k_best)]

    save_table("topic_silhouette", rows)
    save_table("topic_clusters", [
        {"cluster": c, "size": int(cross[c].sum()),
         **{f"n_{s}": int(cross[c][j]) for j, s in enumerate(sources)},
         "top_terms": top_terms[c]}
        for c in range(k_best)])

    # ---- figure ----------------------------------------------------------
    P = TruncatedSVD(n_components=2, random_state=CFG.seed).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot([r["k"] for r in rows], [r["silhouette"] for r in rows], "o-")
    axes[0].axvline(k_best, color="crimson", ls="--", lw=1,
                    label=f"best k = {k_best}")
    axes[0].set_xlabel("number of clusters (k)")
    axes[0].set_ylabel("silhouette score")
    axes[0].set_title("Choosing k by silhouette")
    axes[0].legend()
    axes[0].grid(alpha=.3)

    sc = axes[1].scatter(P[:, 0], P[:, 1], c=km_best.labels_, s=3, cmap="tab20", alpha=.6)
    axes[1].set_title(f"Passage space, k={k_best}  "
                      f"(AMI gold vs distractor = {ami_gold_vs_wiki:.3f})")
    axes[1].set_xlabel("SVD component 1")
    axes[1].set_ylabel("SVD component 2")
    fig.colorbar(sc, ax=axes[1], label="cluster")
    fig.suptitle(f"Unsupervised structure of the {size // 1000}k index", fontsize=12)
    fig.tight_layout()
    out = FIGURES / f"topics_{size // 1000}k.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  figure -> {out.name}")
    verdict = ("gold is NOT separable from its distractors - no distribution artefact"
               if ami_gold_vs_wiki < 0.02 else
               "gold IS separable from its distractors - Recall@k may be inflated")
    print(f"  AMI(cluster, source)        {ami_source:.4f}  "
          f"(NCTB vs Wikipedia is expected to be separable)")
    print(f"  AMI(cluster, gold vs wiki)  {ami_gold_vs_wiki:.4f}  <- {verdict}")
    print(f"  (raw 3-source purity {purity:.3f}, reported for completeness)")

    payload = {"size": size, "sampled": len(subset), "k_best": k_best,
               "silhouette_best": round(sil_best, 4),
               "silhouette_sweep": rows, "source_purity": round(purity, 4),
               "ami_source": round(ami_source, 4),
               "ami_gold_vs_wiki": round(ami_gold_vs_wiki, 4),
               "distribution_artefact": bool(ami_gold_vs_wiki >= 0.02),
               "sources": sources}
    log_result("t7_topics", f"{size // 1000}k", payload)
    return payload


if __name__ == "__main__":
    run()

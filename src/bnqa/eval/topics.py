"""The unsupervised component: K-Means + silhouette over TF-IDF (task T7).

PLAN.md §7.4 answers Sir's Q3 ("where is the unsupervised learning?") with one
figure rather than a paragraph.  K-Means is fitted over the TF-IDF passage
vectors of the headline index for several ``k``, the silhouette coefficient
picks one, and the chosen clustering is described by its top terms per cluster
and by the source mix inside each cluster.

The source mix is the part worth reading.  If the clusters simply separated
Wikipedia from NCTB the figure would be measuring our own index construction
rather than the corpus's topical structure — which is the same distribution
artefact PLAN.md §6.2B warns about, now visible in a second place.
"""

from __future__ import annotations

import numpy as np

from ..config import CFG, FIGURES
from ..retrieval.index import prepare
from ..retrieval.tfidf import TfidfRetriever
from ..utils import log_result, save_table, set_seed


def _sample(passages: list[dict], n: int, seed: int) -> list[dict]:
    if len(passages) <= n:
        return passages
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(passages), size=n, replace=False)
    return [passages[i] for i in sorted(idx)]


def build(size: int | None = None) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    set_seed()
    size = CFG.headline_index if size is None else size
    print(f"T7  topic structure (K-Means + silhouette, {size // 1000}k index)")

    passages = prepare(size)
    sample = _sample(passages, CFG.kmeans_sample, CFG.seed)
    print(f"  {len(sample):,d} passages sampled for clustering")

    vec = TfidfRetriever("word").build(sample)
    X = vec.matrix

    rows: list[dict] = []
    best: tuple[int, float, KMeans] | None = None
    for k in CFG.kmeans_clusters:
        km = KMeans(n_clusters=k, random_state=CFG.seed, n_init=10).fit(X)
        sil = float(silhouette_score(X, km.labels_, sample_size=min(4000, X.shape[0]),
                                     random_state=CFG.seed))
        rows.append({"k": k, "silhouette": round(sil, 4), "inertia": round(float(km.inertia_), 2)})
        print(f"    k={k:3d}  silhouette={sil:.4f}  inertia={km.inertia_:.1f}")
        if best is None or sil > best[1]:
            best = (k, sil, km)

    assert best is not None
    k_best, sil_best, km = best
    print(f"  best k = {k_best} (silhouette {sil_best:.4f})")

    # ---- describe the chosen clustering ----------------------------------
    terms = np.asarray(vec.vectorizer.get_feature_names_out())
    centroids = km.cluster_centers_
    describe: list[dict] = []
    for c in range(k_best):
        top = terms[np.argsort(-centroids[c])[:8]]
        members = [sample[i] for i in np.flatnonzero(km.labels_ == c)]
        mix: dict[str, int] = {}
        for p in members:
            mix[p.get("source") or "?"] = mix.get(p.get("source") or "?", 0) + 1
        dominant = max(mix.items(), key=lambda kv: kv[1]) if mix else ("?", 0)
        describe.append({
            "cluster": c, "size": len(members),
            "top_terms": " · ".join(top),
            "dominant_source": dominant[0],
            "dominant_source_pct": round(100 * dominant[1] / max(len(members), 1), 1),
            **{f"n_{s}": n for s, n in sorted(mix.items())},
        })
    for row in sorted(describe, key=lambda r: -r["size"])[:8]:
        print(f"    c{row['cluster']:<3d} n={row['size']:5,d}  "
              f"{row['dominant_source']:14s} {row['dominant_source_pct']:5.1f}%  "
              f"{row['top_terms']}")

    purity = float(np.mean([r["dominant_source_pct"] for r in describe]))
    print(f"  mean source purity {purity:.1f}% — high purity would mean the clusters "
          f"are separating our index strata, not topics")

    save_table("topics_silhouette", rows)
    save_table("topics_clusters", describe)
    path = _figure(rows, describe, k_best, size)

    payload = {"index_size": size, "sampled": len(sample), "silhouette": rows,
               "best_k": k_best, "best_silhouette": round(sil_best, 4),
               "mean_source_purity_pct": round(purity, 2), "figure": str(path)}
    log_result("t7_topics", f"{size}", payload)
    return payload


def _figure(rows: list[dict], describe: list[dict], k_best: int, size: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ks = [r["k"] for r in rows]
    ax1.plot(ks, [r["silhouette"] for r in rows], "o-", color="#2b6cb0")
    ax1.axvline(k_best, color="#c53030", ls="--", lw=1,
                label=f"chosen k = {k_best}")
    ax1.set_xlabel("number of clusters (k)")
    ax1.set_ylabel("silhouette coefficient")
    ax1.set_title(f"Cluster quality, {size // 1000}k index")
    ax1.legend()
    ax1.grid(alpha=0.3)

    top = sorted(describe, key=lambda r: -r["size"])[:12]
    ax2.barh([f"c{r['cluster']}" for r in top][::-1],
             [r["size"] for r in top][::-1], color="#2f855a")
    ax2.set_xlabel("passages")
    ax2.set_title(f"Cluster sizes (k = {k_best})")
    ax2.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    path = FIGURES / f"topics_{size // 1000}k.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  figure -> {path}")
    return path


if __name__ == "__main__":
    build()

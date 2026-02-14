"""
BERTopic vs OpenAI Embedding Clustering Benchmark
Compare 4 experiment configs on Bitext Customer Support Dataset.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

load_dotenv()

SEED = 42
N_SAMPLES = 2000
EVAL_DIR = Path(__file__).parent


def load_data():
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(ds), N_SAMPLES, replace=False)
    subset = ds.select(indices)
    texts = subset["instruction"]
    labels = subset["intent"]
    return texts, labels


def get_openai_embeddings(texts: list[str]) -> np.ndarray:
    client = OpenAI()
    # API supports up to 2048 per call; we have 2000 so one batch is fine
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([d.embedding for d in response.data])


def get_st_embeddings(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True)


def evaluate(labels_true, labels_pred):
    # Filter out outliers (-1) for ARI/NMI/V but report outlier ratio
    mask = np.array(labels_pred) != -1
    n_outliers = int((~mask).sum())
    outlier_ratio = n_outliers / len(labels_pred)
    n_clusters = len(set(labels_pred) - {-1})

    if mask.sum() < 2:
        return {"ARI": 0, "NMI": 0, "V-Measure": 0, "n_clusters": n_clusters,
                "outlier_ratio": outlier_ratio}

    lt = np.array(labels_true)[mask]
    lp = np.array(labels_pred)[mask]
    return {
        "ARI": round(adjusted_rand_score(lt, lp), 4),
        "NMI": round(normalized_mutual_info_score(lt, lp), 4),
        "V-Measure": round(v_measure_score(lt, lp), 4),
        "n_clusters": n_clusters,
        "outlier_ratio": round(outlier_ratio, 4),
    }


def run_a1(texts, labels):
    """A1: BERTopic with default sentence-transformers."""
    from bertopic import BERTopic
    t0 = time.time()
    model = BERTopic(verbose=False)
    topics, _ = model.fit_transform(texts)
    elapsed = time.time() - t0
    return {"name": "A1: ST + BERTopic", **evaluate(labels, topics), "time_s": round(elapsed, 1)}


def run_a2(texts, labels, embeddings):
    """A2: BERTopic with pre-computed OpenAI embeddings."""
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    t0 = time.time()
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric="euclidean", prediction_data=True)
    model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=False)
    topics, _ = model.fit_transform(texts, embeddings=embeddings)
    elapsed = time.time() - t0
    return {"name": "A2: OpenAI + BERTopic", **evaluate(labels, topics), "time_s": round(elapsed, 1)}


def run_b1(texts, labels, embeddings):
    """B1: OpenAI embeddings + HDBSCAN (no UMAP)."""
    import hdbscan
    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric="euclidean")
    cluster_labels = clusterer.fit_predict(embeddings)
    elapsed = time.time() - t0
    return {"name": "B1: OpenAI + HDBSCAN", **evaluate(labels, cluster_labels), "time_s": round(elapsed, 1)}


def run_b2(texts, labels, embeddings):
    """B2: OpenAI embeddings + K-Means (K=27)."""
    from sklearn.cluster import KMeans
    t0 = time.time()
    km = KMeans(n_clusters=27, random_state=SEED, n_init=10)
    cluster_labels = km.fit_predict(embeddings)
    elapsed = time.time() - t0
    return {"name": "B2: OpenAI + KMeans(27)", **evaluate(labels, cluster_labels), "time_s": round(elapsed, 1)}


def main():
    print("Loading data...")
    texts, labels = load_data()
    print(f"Loaded {len(texts)} samples, {len(set(labels))} unique intents")

    # Get embeddings in parallel
    print("Computing embeddings (ST + OpenAI in parallel)...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_oai = pool.submit(get_openai_embeddings, texts)
        fut_st = pool.submit(get_st_embeddings, texts)
        # Also start A1 in parallel with embedding computation
        # Actually A1 computes its own embeddings, so let's get OAI first
        oai_emb = fut_oai.result()
        st_emb = fut_st.result()

    print(f"OpenAI embeddings: {oai_emb.shape}, ST embeddings: {st_emb.shape}")

    # Run experiments - A1 can run in parallel with A2/B1/B2
    print("\nRunning experiments...")
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(run_a1, texts, labels): "A1",
            pool.submit(run_a2, texts, labels, oai_emb): "A2",
            pool.submit(run_b1, texts, labels, oai_emb): "B1",
            pool.submit(run_b2, texts, labels, oai_emb): "B2",
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  ✓ {r['name']} done ({r['time_s']}s)")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")
                import traceback; traceback.print_exc()

    # Sort by experiment name
    results.sort(key=lambda x: x["name"])

    # Print table
    print("\n" + "=" * 90)
    print(f"{'Experiment':<25} {'ARI':>6} {'NMI':>6} {'V-Meas':>7} {'Clusters':>8} {'Outlier%':>9} {'Time':>6}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<25} {r['ARI']:>6.4f} {r['NMI']:>6.4f} {r['V-Measure']:>7.4f} "
              f"{r['n_clusters']:>8} {r['outlier_ratio']*100:>8.1f}% {r['time_s']:>5.1f}s")
    print("=" * 90)

    # Save
    out_path = EVAL_DIR / "embedding_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

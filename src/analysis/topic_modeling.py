"""
Topic Modeling on Tucson nightlife reviews using BERTopic.
Uses sklearn HDBSCAN (since standalone hdbscan doesn't build on Python 3.14).
UMAP random_state=42 for reproducibility.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

# Load data
print("Loading data...")
df = pd.read_csv(CSV_PATH)
docs = df["text"].fillna("").tolist()
print(f"  {len(docs)} reviews loaded")

# Embedding model
print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# UMAP for dimensionality reduction — random_state=42 for reproducibility
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

# sklearn HDBSCAN for clustering
hdbscan_model = HDBSCAN(
    min_cluster_size=15,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
)

# Vectorizer for topic representation
vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
)

# Build BERTopic
print("Fitting BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10,
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs)

# Results
print(f"\n{'='*60}")
print("TOPIC MODELING RESULTS")
print(f"{'='*60}")

topic_info = topic_model.get_topic_info()
print(f"\nTotal topics found: {len(topic_info) - 1} (excluding outlier topic -1)")
print(f"Outlier reviews: {(np.array(topics) == -1).sum()}")
print(f"\nTopic Info:")
print(topic_info.to_string(index=False))

# Print top words per topic
print(f"\n{'='*60}")
print("TOP WORDS PER TOPIC")
print(f"{'='*60}")
for topic_id in sorted(set(topics)):
    if topic_id == -1:
        continue
    words = topic_model.get_topic(topic_id)
    word_str = ", ".join([w for w, _ in words[:10]])
    count = (np.array(topics) == topic_id).sum()
    print(f"\nTopic {topic_id} ({count} reviews): {word_str}")

# Add topic column to CSV
df["topic"] = topics
df["topic_label"] = df["topic"].map(
    lambda t: "Outlier" if t == -1 else f"Topic {t}"
)
df.to_csv(CSV_PATH, index=False, encoding="utf-8")
print(f"\nSaved topic columns to CSV ({len(df.columns)} columns)")

# --- Visualization 1: Topic sizes bar chart ---
topic_counts = df[df["topic"] != -1]["topic_label"].value_counts().sort_index()
fig1, ax1 = plt.subplots(figsize=(10, 6))
topic_counts.plot(kind="bar", ax=ax1, color="#1f77b4", edgecolor="white")
ax1.set_xlabel("Topic", fontsize=12)
ax1.set_ylabel("Number of Reviews", fontsize=12)
ax1.set_title("Number of Reviews per Topic", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
for i, v in enumerate(topic_counts):
    ax1.text(i, v + 3, str(v), ha="center", fontsize=9)
plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "topic_sizes.png")
plt.savefig(fig1_path, dpi=150)
print(f"Saved: {fig1_path}")

# --- Visualization 2: Top words per topic ---
non_outlier_topics = sorted([t for t in set(topics) if t != -1])
n_topics = len(non_outlier_topics)
if n_topics > 0:
    cols = min(3, n_topics)
    rows = (n_topics + cols - 1) // cols
    fig2, axes2 = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if n_topics == 1:
        axes2 = np.array([axes2])
    axes2 = axes2.flatten()

    for i, topic_id in enumerate(non_outlier_topics):
        ax = axes2[i]
        words_scores = topic_model.get_topic(topic_id)[:10]
        words = [w for w, _ in words_scores]
        scores = [s for _, s in words_scores]
        ax.barh(words[::-1], scores[::-1], color="#2ca02c", edgecolor="white")
        count = (np.array(topics) == topic_id).sum()
        ax.set_title(f"Topic {topic_id} ({count} reviews)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Weight")

    for j in range(i + 1, len(axes2)):
        axes2[j].set_visible(False)

    plt.suptitle("Top Words per Topic", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "topic_top_words.png")
    plt.savefig(fig2_path, dpi=150)
    print(f"Saved: {fig2_path}")

plt.show()

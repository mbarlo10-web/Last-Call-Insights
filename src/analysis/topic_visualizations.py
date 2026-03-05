"""
Advanced BERTopic visualizations:
1. Intertopic Distance Map
2. Topic Hierarchy
3. Topics Over Time
4. Topics per Sentiment Class
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

# Rebuild the model with same parameters for reproducibility
print("Loading sentence-transformers model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0,
    metric="cosine", random_state=42,
)
hdbscan_model = HDBSCAN(
    min_cluster_size=15, min_samples=5,
    metric="euclidean", cluster_selection_method="eom",
)
vectorizer_model = CountVectorizer(
    stop_words="english", ngram_range=(1, 2), min_df=3,
)

print("Fitting BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10,
    verbose=False,
)
topics, probs = topic_model.fit_transform(docs)
print(f"  {len(set(topics)) - 1} topics found")

# ============================================================
# 1. Intertopic Distance Map
# ============================================================
print("\n1. Generating Intertopic Distance Map...")
fig_distance = topic_model.visualize_topics()
dist_path = os.path.join(FIG_DIR, "intertopic_distance_map.html")
fig_distance.write_html(dist_path)
print(f"   Saved: {dist_path}")

# Also save as static image
fig_distance.write_image(os.path.join(FIG_DIR, "intertopic_distance_map.png"), width=1000, height=800, scale=2)
print(f"   Saved: {os.path.join(FIG_DIR, 'intertopic_distance_map.png')}")

# ============================================================
# 2. Topic Hierarchy
# ============================================================
print("\n2. Generating Topic Hierarchy...")
fig_hierarchy = topic_model.visualize_hierarchy()
hier_path = os.path.join(FIG_DIR, "topic_hierarchy.html")
fig_hierarchy.write_html(hier_path)
print(f"   Saved: {hier_path}")

fig_hierarchy.write_image(os.path.join(FIG_DIR, "topic_hierarchy.png"), width=1200, height=800, scale=2)
print(f"   Saved: {os.path.join(FIG_DIR, 'topic_hierarchy.png')}")

# ============================================================
# 3. Topics Over Time
# ============================================================
print("\n3. Generating Topics Over Time...")
df["date_parsed"] = pd.to_datetime(df["date"], format="%m/%d/%y")
timestamps = df["date_parsed"].tolist()

topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
fig_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
time_path = os.path.join(FIG_DIR, "topics_over_time.html")
fig_time.write_html(time_path)
print(f"   Saved: {time_path}")

fig_time.write_image(os.path.join(FIG_DIR, "topics_over_time.png"), width=1400, height=600, scale=2)
print(f"   Saved: {os.path.join(FIG_DIR, 'topics_over_time.png')}")

# ============================================================
# 4. Topics per Sentiment Class
# ============================================================
print("\n4. Generating Topics per Sentiment Class...")
classes = df["sentiment_label"].tolist()

topics_per_class = topic_model.topics_per_class(docs, classes=classes)
fig_class = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10)
class_path = os.path.join(FIG_DIR, "topics_per_sentiment.html")
fig_class.write_html(class_path)
print(f"   Saved: {class_path}")

fig_class.write_image(os.path.join(FIG_DIR, "topics_per_sentiment.png"), width=1400, height=800, scale=2)
print(f"   Saved: {os.path.join(FIG_DIR, 'topics_per_sentiment.png')}")

print("\nAll visualizations saved!")

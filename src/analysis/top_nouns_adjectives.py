"""Bar charts for top 20 nouns and top 20 adjectives from POS-tagged reviews."""

import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
ADJ_TAGS = {"JJ", "JJR", "JJS"}

df = pd.read_csv(CSV_PATH)

# Parse pos_tags column: "great_JJ|place_NN|..." -> list of (word, tag)
noun_counter = Counter()
adj_counter = Counter()

for row in df["pos_tags"].dropna():
    for pair in row.split("|"):
        if "_" in pair:
            word, tag = pair.rsplit("_", 1)
            if tag in NOUN_TAGS:
                noun_counter[word] += 1
            elif tag in ADJ_TAGS:
                adj_counter[word] += 1

top_nouns = noun_counter.most_common(20)
top_adjs = adj_counter.most_common(20)

# --- Plot Top 20 Nouns ---
fig1, ax1 = plt.subplots(figsize=(10, 7))
words, counts = zip(*top_nouns)
ax1.barh(words, counts, color="#2ca02c", edgecolor="white")
ax1.set_xlabel("Frequency", fontsize=12)
ax1.set_title("Top 20 Nouns in Nightlife Reviews", fontsize=14, fontweight="bold")
ax1.invert_yaxis()
for i, v in enumerate(counts):
    ax1.text(v + 5, i, str(v), va="center", fontsize=9)
plt.tight_layout()
noun_path = os.path.join(FIG_DIR, "top_nouns.png")
plt.savefig(noun_path, dpi=150)
print(f"Saved: {noun_path}")

# --- Plot Top 20 Adjectives ---
fig2, ax2 = plt.subplots(figsize=(10, 7))
words, counts = zip(*top_adjs)
ax2.barh(words, counts, color="#d62728", edgecolor="white")
ax2.set_xlabel("Frequency", fontsize=12)
ax2.set_title("Top 20 Adjectives in Nightlife Reviews", fontsize=14, fontweight="bold")
ax2.invert_yaxis()
for i, v in enumerate(counts):
    ax2.text(v + 5, i, str(v), va="center", fontsize=9)
plt.tight_layout()
adj_path = os.path.join(FIG_DIR, "top_adjectives.png")
plt.savefig(adj_path, dpi=150)
print(f"Saved: {adj_path}")

plt.show()

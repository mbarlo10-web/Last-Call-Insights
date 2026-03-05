"""Unhappiness Index for Tucson nightlife venues.
Reviews with sentiment_score < 0.2 are flagged as 'unhappy'.
Unhappiness Index = unhappy_rate * unhappy_count
(captures both proportion and volume of unhappy reviews)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

df = pd.read_csv(CSV_PATH)

# Flag unhappy reviews
df["is_unhappy"] = (df["sentiment_score"] < 0.2).astype(int)

# Per-business aggregation
biz = df.groupby("name").agg(
    total_reviews=("sentiment_score", "count"),
    avg_sentiment=("sentiment_score", "mean"),
    unhappy_count=("is_unhappy", "sum"),
).reset_index()

biz["unhappy_rate"] = biz["unhappy_count"] / biz["total_reviews"]
biz["unhappiness_index"] = (biz["unhappy_rate"] * biz["unhappy_count"]).round(2)
biz = biz.sort_values("unhappiness_index", ascending=True)

# Plot — all 39 venues
fig, ax = plt.subplots(figsize=(12, 14))
colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(biz)))
bars = ax.barh(biz["name"], biz["unhappiness_index"], color=colors, edgecolor="white")

ax.set_xlabel("Unhappiness Index", fontsize=12)
ax.set_ylabel("")
ax.set_title("Unhappiness Index by Venue\n(sentiment_score < 0.2 threshold)", fontsize=14, fontweight="bold")

for i, (idx, row) in enumerate(biz.iterrows()):
    label = f'{row["unhappiness_index"]:.1f}  ({row["unhappy_count"]}/{row["total_reviews"]})'
    ax.text(row["unhappiness_index"] + 0.2, i, label, va="center", fontsize=8)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "unhappiness_index.png")
plt.savefig(fig_path, dpi=150)
print(f"Saved: {fig_path}")

plt.show()

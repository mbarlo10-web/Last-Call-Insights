"""Venues ranked by gap between review rating and sentiment score.
Normalizes rating (1-5) to -1 to +1 scale for fair comparison with sentiment (-1 to 1).
Rating mapping: 1→-1, 3→0, 5→+1  =>  (rating - 3) / 2
Large positive gap = rating higher than sentiment (over-rated by stars).
Large negative gap = sentiment higher than rating (under-rated by stars).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

df = pd.read_csv(CSV_PATH)

biz = df.groupby("name").agg(
    avg_rating=("rating", "mean"),
    avg_sentiment=("sentiment_score", "mean"),
    total_reviews=("sentiment_score", "count"),
).reset_index()

# Normalize rating to -1 to +1 scale: (rating - 3) / 2
# 1→-1, 2→-0.5, 3→0, 4→+0.5, 5→+1
biz["rating_normalized"] = (biz["avg_rating"] - 3) / 2
biz["gap"] = (biz["rating_normalized"] - biz["avg_sentiment"]).round(3)
biz["abs_gap"] = biz["gap"].abs()
biz = biz.sort_values("abs_gap", ascending=True)

# Plot
fig, ax = plt.subplots(figsize=(12, 14))
colors = ["#d62728" if g > 0 else "#2ca02c" for g in biz["gap"]]
ax.barh(biz["name"], biz["gap"], color=colors, edgecolor="white")
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Gap (Normalized Rating - Avg Sentiment)", fontsize=12)
ax.set_ylabel("")
ax.set_title("Rating vs Sentiment Gap by Venue\nRed = Rating > Sentiment | Green = Sentiment > Rating",
             fontsize=14, fontweight="bold")

for i, (_, row) in enumerate(biz.iterrows()):
    label = f'{row["gap"]:+.2f}  (Rating: {row["avg_rating"]:.1f}, Sent: {row["avg_sentiment"]:.2f})'
    x_pos = row["gap"] + (0.01 if row["gap"] >= 0 else -0.01)
    ha = "left" if row["gap"] >= 0 else "right"
    ax.text(x_pos, i, label, va="center", ha=ha, fontsize=7)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "rating_sentiment_gap.png")
plt.savefig(fig_path, dpi=150)
print(f"Saved: {fig_path}")

# Print table
print("\nTop venues by absolute gap:")
top = biz.sort_values("abs_gap", ascending=False)
print(top[["name", "total_reviews", "avg_rating", "avg_sentiment", "rating_normalized", "gap"]].round(3).head(15).to_string(index=False))

plt.show()

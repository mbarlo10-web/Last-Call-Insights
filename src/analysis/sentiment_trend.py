"""Line chart showing sentiment trends over time (by quarter)."""

import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

df = pd.read_csv(CSV_PATH)

# Count reviews per quarter per sentiment
ct = pd.crosstab(df["quarter"], df["sentiment_label"])
for s in ["Positive", "Neutral", "Negative"]:
    if s not in ct.columns:
        ct[s] = 0
ct = ct[["Positive", "Neutral", "Negative"]].sort_index()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ct.index, ct["Positive"], color="#2ca02c", marker="o", linewidth=2, label="Positive")
ax.plot(ct.index, ct["Neutral"], color="#ff7f0e", marker="s", linewidth=2, label="Neutral")
ax.plot(ct.index, ct["Negative"], color="#d62728", marker="^", linewidth=2, label="Negative")

ax.set_xlabel("Quarter", fontsize=12)
ax.set_ylabel("Number of Reviews", fontsize=12)
ax.set_title("Sentiment Trend Over Time (by Quarter)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "sentiment_trend_quarter.png")
plt.savefig(fig_path, dpi=150)
print(f"Saved: {fig_path}")

plt.show()

"""Grouped bar chart: sentiment breakdown per aspect dimension."""

import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

df = pd.read_csv(CSV_PATH)

ct = pd.crosstab(df["aspect_dimension"], df["sentiment_label"])
for s in ["Positive", "Neutral", "Negative"]:
    if s not in ct.columns:
        ct[s] = 0
ct = ct[["Positive", "Neutral", "Negative"]]
ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(12, 6))
ct.plot(kind="barh", ax=ax, color=["#2ca02c", "#ff7f0e", "#d62728"], edgecolor="white")
ax.set_xlabel("Number of Reviews", fontsize=12)
ax.set_ylabel("")
ax.set_title("Sentiment Breakdown by Aspect Dimension", fontsize=14, fontweight="bold")
ax.legend(title="Sentiment", fontsize=11)
ax.invert_yaxis()
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "sentiment_by_aspect_dimension.png")
plt.savefig(fig_path, dpi=150)
print(f"Saved: {fig_path}")

# Print the crosstab
print("\n" + ct.to_string())

plt.show()

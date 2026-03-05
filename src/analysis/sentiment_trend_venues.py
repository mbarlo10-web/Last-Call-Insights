"""Sentiment trend over time for venues with the biggest sentiment changes."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

df = pd.read_csv(CSV_PATH)
df["year"] = pd.to_datetime(df["date"], format="%m/%d/%y").dt.year

# Find venues with biggest sentiment change (need 10+ reviews, 3+ years)
results = []
for name, grp in df.groupby("name"):
    if len(grp) < 10:
        continue
    yearly = grp.groupby("year")["sentiment_score"].mean()
    if len(yearly) < 3:
        continue
    slope, _ = np.polyfit(yearly.index, yearly.values, 1)
    results.append({"name": name, "slope": slope, "reviews": len(grp)})

res = pd.DataFrame(results).sort_values("slope")

# Top 4 declining + Top 4 improving
declining = res.head(4)["name"].tolist()
improving = res.tail(4)["name"].tolist()
selected = declining + improving

# Plot
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
fig.suptitle("Sentiment Trend Over Time — Biggest Changers", fontsize=16, fontweight="bold", y=1.02)

for i, name in enumerate(selected):
    row = i // 4
    col = i % 4
    ax = axes[row][col]

    grp = df[df["name"] == name]
    yearly = grp.groupby("year")["sentiment_score"].mean()

    color = "#d62728" if name in declining else "#2ca02c"
    ax.plot(yearly.index, yearly.values, marker="o", color=color, linewidth=2, markersize=6)

    # Trend line
    slope, intercept = np.polyfit(yearly.index, yearly.values, 1)
    trend_x = np.array([yearly.index.min(), yearly.index.max()])
    ax.plot(trend_x, slope * trend_x + intercept, "--", color=color, alpha=0.5, linewidth=1.5)

    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    direction = "Declining" if name in declining else "Improving"
    ax.set_title(f"{name}\n({direction}, slope={slope:+.3f}, n={len(grp)})", fontsize=9, fontweight="bold")
    ax.set_xlabel("Year", fontsize=9)
    if col == 0:
        ax.set_ylabel("Avg Sentiment Score", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

axes[0][0].set_ylabel("Avg Sentiment Score\n(Declining)", fontsize=10, color="#d62728")
axes[1][0].set_ylabel("Avg Sentiment Score\n(Improving)", fontsize=10, color="#2ca02c")

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "sentiment_trend_venues.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path}")

# Print summary
print("\nDECLINING venues:")
for name in declining:
    grp = df[df["name"] == name]
    yearly = grp.groupby("year")["sentiment_score"].mean()
    slope, _ = np.polyfit(yearly.index, yearly.values, 1)
    print(f"  {name}: slope={slope:+.4f}, reviews={len(grp)}")

print("\nIMPROVING venues:")
for name in improving:
    grp = df[df["name"] == name]
    yearly = grp.groupby("year")["sentiment_score"].mean()
    slope, _ = np.polyfit(yearly.index, yearly.values, 1)
    print(f"  {name}: slope={slope:+.4f}, reviews={len(grp)}")

plt.show()

"""Bar charts for frequency of core_venue_type, food_emphasis, and experience_context."""

import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

df = pd.read_csv(CSV_PATH)

columns = ["core_venue_type", "food_emphasis", "experience_context"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
titles = [
    "Frequency by Core Venue Type",
    "Frequency by Food Emphasis",
    "Frequency by Experience Context",
]

for col, color, title in zip(columns, colors, titles):
    counts = df[col].value_counts()

    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.5 + 1)))
    counts.plot(kind="barh", ax=ax, color=color, edgecolor="white")
    ax.set_xlabel("Number of Reviews", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for i, v in enumerate(counts):
        ax.text(v + 5, i, str(v), va="center", fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"{col}_frequency.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")
    print(counts.to_string())
    print()

plt.show()

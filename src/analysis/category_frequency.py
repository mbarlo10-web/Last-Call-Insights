"""Bar chart showing frequency of each category across Tucson nightlife venues."""

import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")
FIG_PATH = os.path.join(PROJECT_ROOT, "outputs", "figures", "category_frequency.png")

df = pd.read_csv(CSV_PATH)

# Each row's full categories string is treated as one category
cat_counts = df["categories"].value_counts()

print(f"Total unique categories: {len(cat_counts)}")
print(cat_counts.to_string())

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
cat_counts.plot(kind="barh", ax=ax, color="#1f77b4", edgecolor="white")
ax.set_xlabel("Number of Venues", fontsize=12)
ax.set_ylabel("")
ax.set_title("Category Frequency Across Tucson Nightlife Venues", fontsize=14, fontweight="bold")
ax.invert_yaxis()

for i, v in enumerate(cat_counts):
    ax.text(v + 0.3, i, str(v), va="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=150)
print(f"\nChart saved to: {FIG_PATH}")
plt.show()

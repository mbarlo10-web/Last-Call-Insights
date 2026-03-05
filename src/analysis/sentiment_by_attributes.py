"""Compare sentiment across venue attributes with grouped bar charts."""

import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

df = pd.read_csv(CSV_PATH)

# --- Clean attribute columns ---

# NoiseLevel: strip u'' and '' wrappers
df["noise_level"] = df["attributes.NoiseLevel"].astype(str).str.replace("u'", "", regex=False).str.replace("'", "", regex=False).str.strip()
df.loc[df["noise_level"].isin(["nan", ""]), "noise_level"] = np.nan

# Alcohol: strip u'' and '' wrappers
df["alcohol"] = df["attributes.Alcohol"].astype(str).str.replace("u'", "", regex=False).str.replace("'", "", regex=False).str.strip()
df.loc[df["alcohol"].isin(["nan", ""]), "alcohol"] = np.nan

# Binary columns: map 1/0 to Yes/No
binary_cols = {
    "attributes.HappyHour": "happy_hour",
    "attributes.GoodForDancing": "good_for_dancing",
    "attributes.HasTV": "has_tv",
    "attributes.OutdoorSeating": "outdoor_seating",
}
for orig, clean in binary_cols.items():
    df[clean] = df[orig].map({1.0: "Yes", 0.0: "No"})

# Ambience: extract key traits (casual, classy, trendy, divey, hipster)
def parse_ambience(val):
    if pd.isna(val) or val == "nan":
        return np.nan
    try:
        d = ast.literal_eval(val.replace("u'", "'").replace("None", "None").replace("True", "True").replace("False", "False"))
        traits = [k for k, v in d.items() if v is True]
        return ", ".join(traits) if traits else "none"
    except:
        return np.nan

df["ambience"] = df["attributes.Ambience"].apply(parse_ambience)


def plot_sentiment_comparison(df, col, title, fig_name):
    """Create a grouped bar chart: sentiment label counts per attribute value."""
    subset = df.dropna(subset=[col])
    if subset.empty:
        return

    ct = pd.crosstab(subset[col], subset["sentiment_label"])
    # Ensure all sentiment columns exist
    for s in ["Positive", "Neutral", "Negative"]:
        if s not in ct.columns:
            ct[s] = 0
    ct = ct[["Positive", "Neutral", "Negative"]]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(ct) * 0.6 + 1)))
    ct.plot(kind="barh", ax=ax, color=["#2ca02c", "#ff7f0e", "#d62728"], edgecolor="white")
    ax.set_xlabel("Number of Reviews", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title="Sentiment")
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(FIG_DIR, fig_name)
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")


# Generate all charts
plot_sentiment_comparison(df, "noise_level", "Sentiment by Noise Level", "sentiment_noise_level.png")
plot_sentiment_comparison(df, "alcohol", "Sentiment by Alcohol Type", "sentiment_alcohol.png")
plot_sentiment_comparison(df, "happy_hour", "Sentiment by Happy Hour", "sentiment_happy_hour.png")
plot_sentiment_comparison(df, "has_tv", "Sentiment by Has TV", "sentiment_has_tv.png")
plot_sentiment_comparison(df, "outdoor_seating", "Sentiment by Outdoor Seating", "sentiment_outdoor_seating.png")
plot_sentiment_comparison(df, "good_for_dancing", "Sentiment by Good For Dancing", "sentiment_good_for_dancing.png")
plot_sentiment_comparison(df, "ambience", "Sentiment by Ambience", "sentiment_ambience.png")

plt.show()

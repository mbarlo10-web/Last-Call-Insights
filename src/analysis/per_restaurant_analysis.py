"""
Per-restaurant analysis for Barrio Brewing, Penca, and Blackrock Brewers.

Generates 3 dashboards per restaurant:
  1. Review Overview: total reviews, avg star rating, review volume trend
  2. Sentiment Analysis: avg sentiment score, % positive/neutral/negative, sentiment by dimension
  3. Topic Analysis: top 3 topics, % mentioning each, sentiment per topic, topic trend over time
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "per_restaurant")

RESTAURANTS = {
    "Barrio Brewing": "Barrio_Brewing",
    "Penca": "Penca",
    "Blackrock Brewers": "Blackrock_Brewers",
}

SENTIMENT_COLORS = {"Positive": "#2ca02c", "Neutral": "#ff7f0e", "Negative": "#d62728"}
TOPIC_COLORS = plt.cm.Set2.colors


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["date_parsed"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    df["year"] = df["date_parsed"].dt.year
    df["year_quarter"] = df["date_parsed"].dt.to_period("Q").astype(str)
    return df


# ── Dashboard 1: Review Overview ─────────────────────────────────────────────

def plot_review_overview(rdf, name, out_folder):
    """Total reviews, avg star rating, review volume trend."""
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2], wspace=0.35)
    fig.suptitle(f"{name} — Review Overview", fontsize=16, fontweight="bold", y=1.02)

    # 1a. Total number of reviews (big number card)
    ax1 = fig.add_subplot(gs[0])
    total = len(rdf)
    ax1.text(0.5, 0.55, str(total), ha="center", va="center",
             fontsize=60, fontweight="bold", color="#1f77b4",
             transform=ax1.transAxes)
    ax1.text(0.5, 0.2, "Total Reviews", ha="center", va="center",
             fontsize=14, color="gray", transform=ax1.transAxes)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    # 1b. Average star rating (big number + star bar)
    ax2 = fig.add_subplot(gs[1])
    avg_rating = rdf["rating"].mean()
    ax2.text(0.5, 0.6, f"{avg_rating:.2f}", ha="center", va="center",
             fontsize=52, fontweight="bold", color="#ff9800",
             transform=ax2.transAxes)
    ax2.text(0.5, 0.35, "★" * round(avg_rating) + "☆" * (5 - round(avg_rating)),
             ha="center", va="center", fontsize=20, color="#ff9800",
             transform=ax2.transAxes)
    ax2.text(0.5, 0.15, "Avg Star Rating", ha="center", va="center",
             fontsize=14, color="gray", transform=ax2.transAxes)
    ax2.axis("off")

    # 1c. Review volume trend (bar chart by year)
    ax3 = fig.add_subplot(gs[2])
    yearly = rdf.groupby("year").size()
    ax3.bar(yearly.index.astype(str), yearly.values, color="#1f77b4", edgecolor="white")
    ymax = yearly.values.max()
    ylim_top = ymax + max(ymax * 0.15, 1)  # ensure room for labels even with small counts
    for i, v in enumerate(yearly.values):
        ax3.text(i, v + ylim_top * 0.02, str(v), ha="center", fontsize=9, fontweight="bold")
    ax3.set_xlabel("Year", fontsize=11)
    ax3.set_ylabel("Number of Reviews", fontsize=11)
    ax3.set_title("Review Volume Trend", fontsize=13, fontweight="bold")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(axis="y", alpha=0.3)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax3.set_ylim(0, ylim_top)

    fig.subplots_adjust(top=0.88)
    path = os.path.join(out_folder, "1_review_overview.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Dashboard 2: Sentiment Analysis ──────────────────────────────────────────

def plot_sentiment_analysis(rdf, name, out_folder):
    """Avg sentiment score, % positive/neutral/negative, sentiment by dimension."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [1, 1.2, 1.8]})
    fig.suptitle(f"{name} — Sentiment Analysis", fontsize=18, fontweight="bold")

    # 2a. Average sentiment score (big number card)
    ax1 = axes[0]
    avg_sent = rdf["sentiment_score"].mean()
    color = "#2ca02c" if avg_sent > 0.4 else ("#ff7f0e" if avg_sent > 0.2 else "#d62728")
    ax1.text(0.5, 0.55, f"{avg_sent:.3f}", ha="center", va="center",
             fontsize=48, fontweight="bold", color=color,
             transform=ax1.transAxes)
    ax1.text(0.5, 0.2, "Avg Sentiment Score", ha="center", va="center",
             fontsize=13, color="gray", transform=ax1.transAxes)
    ax1.axis("off")

    # 2b. % positive, neutral, negative (pie chart)
    ax2 = axes[1]
    counts = rdf["sentiment_label"].value_counts()
    labels_order = ["Positive", "Neutral", "Negative"]
    sizes = [counts.get(l, 0) for l in labels_order]
    total = sum(sizes)
    pcts = [(s / total) * 100 if total > 0 else 0 for s in sizes]
    colors = [SENTIMENT_COLORS[l] for l in labels_order]

    # Only include non-zero slices
    filtered = [(l, s, c, p) for l, s, c, p in zip(labels_order, sizes, colors, pcts) if s > 0]
    f_labels, f_sizes, f_colors, f_pcts = zip(*filtered) if filtered else ([], [], [], [])

    # Build autopct that combines label + pct for each slice
    def make_label(pct):
        return f"{pct:.1f}%" if pct >= 5 else ""

    wedges, texts, autotexts = ax2.pie(
        f_sizes, colors=f_colors, autopct=make_label,
        startangle=90, textprops={"fontsize": 11},
        pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")

    # Add a clean legend instead of crowded labels on slices
    legend_labels = [f"{l}: {p:.1f}% (n={s})" for l, s, p in zip(f_labels, f_sizes, f_pcts)]
    ax2.legend(wedges, legend_labels, loc="lower center", fontsize=9,
               bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax2.set_title("Sentiment Distribution", fontsize=13, fontweight="bold")

    # 2c. Sentiment by aspect dimension (horizontal stacked bar)
    ax3 = axes[2]
    ct = pd.crosstab(rdf["aspect_dimension"], rdf["sentiment_label"])
    for s in ["Positive", "Neutral", "Negative"]:
        if s not in ct.columns:
            ct[s] = 0
    ct = ct[["Positive", "Neutral", "Negative"]]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct = ct_pct.loc[ct.sum(axis=1).sort_values(ascending=True).index]

    left = np.zeros(len(ct_pct))
    for label in ["Positive", "Neutral", "Negative"]:
        vals = ct_pct[label].values
        ax3.barh(ct_pct.index, vals, left=left, color=SENTIMENT_COLORS[label],
                 label=label, edgecolor="white", height=0.6)
        left += vals
    ax3.set_xlabel("% of Reviews", fontsize=11)
    ax3.set_title("Sentiment by Dimension", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10, loc="lower right")
    ax3.set_xlim(0, 105)

    fig.subplots_adjust(top=0.88, wspace=0.4)
    path = os.path.join(out_folder, "2_sentiment_analysis.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Dashboard 3: Topic Analysis ──────────────────────────────────────────────

def plot_topic_analysis(rdf, name, out_folder, full_df):
    """Top 3 topics, % mentioning each, sentiment per topic, topic trend."""
    # Identify top 3 topics (excluding outlier -1)
    valid = rdf[rdf["topic"] != -1]
    if len(valid) == 0:
        print(f"  ⚠ {name}: no non-outlier topics, skipping topic dashboard")
        return

    topic_counts = valid["topic"].value_counts()
    top3_topics = topic_counts.head(3).index.tolist()

    # Build topic labels from the full dataset's most common nouns/adj per topic
    topic_labels = {}
    for t in top3_topics:
        topic_reviews = full_df[full_df["topic"] == t]
        all_tokens = []
        for tok_str in topic_reviews["tokens_nouns_adj"].dropna():
            all_tokens.extend(tok_str.strip("[]").replace("'", "").split(", "))
        common = Counter(all_tokens).most_common(3)
        label = ", ".join([w for w, _ in common if w])
        topic_labels[t] = f"Topic {t}: {label}"

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"{name} — Topic Analysis", fontsize=18, fontweight="bold")

    # Build short x-tick labels with keywords
    xtick_labels = []
    for t in top3_topics:
        kw = topic_labels[t].split(": ", 1)[1] if ": " in topic_labels[t] else ""
        # Truncate long keyword strings
        if len(kw) > 25:
            kw = kw[:22] + "..."
        xtick_labels.append(f"Topic {t}\n({kw})")

    colors_3 = [TOPIC_COLORS[i % len(TOPIC_COLORS)] for i in range(len(top3_topics))]

    # Bar width — narrower when fewer topics so bars don't fill the whole axis
    n_topics = len(top3_topics)
    bar_width = 0.3 if n_topics == 1 else (0.5 if n_topics == 2 else 0.7)
    x_margin = 0.8 if n_topics == 1 else 0.5

    # 3a. Top 3 most mentioned topics (bar chart with counts)
    ax1 = axes[0, 0]
    t_counts = [topic_counts[t] for t in top3_topics]
    ymax1 = max(t_counts)
    ylim1 = ymax1 + max(ymax1 * 0.15, 1)
    ax1.bar(range(n_topics), t_counts, width=bar_width, color=colors_3, edgecolor="white")
    ax1.set_xticks(range(n_topics))
    ax1.set_xticklabels(xtick_labels, fontsize=9)
    ax1.set_xlim(-x_margin, n_topics - 1 + x_margin)
    for i, v in enumerate(t_counts):
        ax1.text(i, v + ylim1 * 0.02, str(v), ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Reviews", fontsize=11)
    ax1.set_title("Top 3 Most Mentioned Topics", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.set_ylim(0, ylim1)

    # 3b. % of reviews mentioning each topic
    ax2 = axes[0, 1]
    total_reviews = len(rdf)
    pcts = [(topic_counts[t] / total_reviews) * 100 for t in top3_topics]
    ymax2 = max(pcts)
    ylim2 = ymax2 + max(ymax2 * 0.15, 5)
    ax2.bar(range(n_topics), pcts, width=bar_width, color=colors_3, edgecolor="white")
    ax2.set_xticks(range(n_topics))
    ax2.set_xticklabels(xtick_labels, fontsize=9)
    ax2.set_xlim(-x_margin, n_topics - 1 + x_margin)
    for i, v in enumerate(pcts):
        ax2.text(i, v + ylim2 * 0.02, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("% of Reviews", fontsize=11)
    ax2.set_title("% of Reviews per Topic", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, ylim2)

    # 3c. Sentiment breakdown per topic (stacked bar)
    ax3 = axes[1, 0]
    topic_sentiment = []
    for t in top3_topics:
        subset = rdf[rdf["topic"] == t]
        for label in ["Positive", "Neutral", "Negative"]:
            cnt = (subset["sentiment_label"] == label).sum()
            topic_sentiment.append({"topic": f"Topic {t}", "sentiment": label, "count": cnt})
    ts_df = pd.DataFrame(topic_sentiment)
    pivot = ts_df.pivot(index="topic", columns="sentiment", values="count").fillna(0)
    for s in ["Positive", "Neutral", "Negative"]:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[["Positive", "Neutral", "Negative"]]
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    left = np.zeros(len(pivot_pct))
    for label in ["Positive", "Neutral", "Negative"]:
        vals = pivot_pct[label].values
        ax3.barh(pivot_pct.index, vals, left=left, color=SENTIMENT_COLORS[label],
                 label=label, edgecolor="white", height=0.5)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 8:
                ax3.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                         fontsize=10, fontweight="bold", color="white")
        left += vals
    ax3.set_xlabel("% of Reviews", fontsize=11)
    ax3.set_title("Sentiment Breakdown per Topic", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10, loc="lower right")
    ax3.set_xlim(0, 105)

    # 3d. Topic trend over time (line chart by year)
    ax4 = axes[1, 1]
    all_years = sorted(rdf["year"].unique())
    for i, t in enumerate(top3_topics):
        subset = rdf[rdf["topic"] == t]
        yearly = subset.groupby("year").size().reindex(all_years, fill_value=0)
        ax4.plot(yearly.index, yearly.values, marker="o",
                 color=TOPIC_COLORS[i % len(TOPIC_COLORS)],
                 linewidth=2.5, markersize=7, label=f"Topic {t}")
    ax4.set_xlabel("Year", fontsize=11)
    ax4.set_ylabel("Number of Reviews", fontsize=11)
    ax4.set_title("Topic Trend Over Time", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.set_xticks(all_years)
    ax4.set_xticklabels([str(y) for y in all_years], rotation=45)
    ax4.grid(axis="y", alpha=0.3)
    ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.subplots_adjust(top=0.92, hspace=0.35, wspace=0.3)
    path = os.path.join(out_folder, "3_topic_analysis.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    for name, folder_name in RESTAURANTS.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        rdf = df[df["name"] == name].copy()
        if len(rdf) == 0:
            print(f"  ⚠ No reviews found for '{name}', skipping.")
            continue

        out_folder = os.path.join(OUT_DIR, folder_name)
        os.makedirs(out_folder, exist_ok=True)

        print(f"  Reviews: {len(rdf)}")
        plot_review_overview(rdf, name, out_folder)
        plot_sentiment_analysis(rdf, name, out_folder)
        plot_topic_analysis(rdf, name, out_folder, df)

    print(f"\nAll dashboards saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

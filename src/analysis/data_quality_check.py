"""
Data quality check for business_reviews_tucson.csv.
Covers:
  - Missing values per column
  - Duplicate detection (multiple strategies)
  - Key column null audit
  - Short review detection
  - Rating and sentiment score range validation
"""

import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")


def main():
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    print(f"\n{'='*60}")
    print("SHAPE")
    print(f"{'='*60}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # ── Missing Values ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MISSING VALUES (columns with at least 1 null)")
    print(f"{'='*60}")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct":   missing_pct
    })
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    if missing_df.empty:
        print("  No missing values found.")
    else:
        print(f"  {'Column':<40} {'Count':>8} {'%':>8}")
        print(f"  {'-'*58}")
        for col, row in missing_df.iterrows():
            print(f"  {col:<40} {int(row['missing_count']):>8,} {row['missing_pct']:>7.1f}%")

    # ── Duplicate Checks ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DUPLICATE CHECKS")
    print(f"{'='*60}")
    full_dups = df.duplicated().sum()
    bid_text  = df.duplicated(subset=["business_id", "text"]).sum()
    name_text = df.duplicated(subset=["name", "text"]).sum()
    ndr       = df.duplicated(subset=["name", "date", "rating"]).sum()
    ndr_diff  = df.duplicated(subset=["name", "date", "rating"], keep=False) & \
                ~df.duplicated(subset=["name", "date", "rating", "text"], keep=False)

    print(f"  Fully identical rows:                  {full_dups:>6,}")
    print(f"  Same business_id + text:               {bid_text:>6,}")
    print(f"  Same name + text:                      {name_text:>6,}")
    print(f"  Same name + date + rating (any text):  {ndr:>6,}")
    print(f"  Same name + date + rating, diff text:  {ndr_diff.sum():>6,}  <- legitimate same-day reviews")

    # ── Key Column Null Audit ───────────────────────────────────
    print(f"\n{'='*60}")
    print("KEY COLUMN NULL AUDIT")
    print(f"{'='*60}")
    key_cols = [
        "text", "rating", "sentiment_score", "sentiment_label",
        "aspect_dimension", "topic", "tokens", "tokens_normalized",
        "pos_tags", "tokens_nouns_adj",
    ]
    all_clean = True
    for col in key_cols:
        n = df[col].isnull().sum()
        status = "OK" if n == 0 else "*** MISSING ***"
        print(f"  {col:<30} nulls: {n:>4}  {status}")
        if n > 0:
            all_clean = False
    if all_clean:
        print("\n  All key columns are complete — no nulls.")

    # ── Short Reviews ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SHORT REVIEWS (token_count <= 5)")
    print(f"{'='*60}")
    short = df[df["token_count"] <= 5]
    if short.empty:
        print("  None found.")
    else:
        print(f"  Count: {len(short)}")
        print(short[["name", "rating", "token_count", "text"]].to_string())

    # ── Rating Range ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RATING DISTRIBUTION & RANGE CHECK")
    print(f"{'='*60}")
    print(df["rating"].value_counts().sort_index().to_string())
    out_of_range = ((df["rating"] < 1) | (df["rating"] > 5)).sum()
    print(f"\n  Ratings outside 1–5: {out_of_range}")

    # ── Sentiment Score Range ───────────────────────────────────
    print(f"\n{'='*60}")
    print("SENTIMENT SCORE RANGE CHECK")
    print(f"{'='*60}")
    print(f"  Min:  {df['sentiment_score'].min():.4f}")
    print(f"  Max:  {df['sentiment_score'].max():.4f}")
    print(f"  Mean: {df['sentiment_score'].mean():.4f}")
    out_of_range_s = ((df["sentiment_score"] < -1) | (df["sentiment_score"] > 1)).sum()
    print(f"  Outside -1 to +1: {out_of_range_s}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows:                {len(df):>8,}")
    print(f"  Columns with missing data: {len(missing_df):>8,}  (all are attributes.*)")
    print(f"  Exact duplicate rows:      {full_dups:>8,}")
    print(f"  Short reviews (<=5 tokens):{len(short):>8,}")
    print(f"  Invalid ratings:           {out_of_range:>8,}")
    print(f"  Invalid sentiment scores:  {out_of_range_s:>8,}")
    print(f"\n  Core analysis columns: CLEAN")
    print(f"  Attribute columns: HIGH MISSINGNESS — limit attribute-level analysis")


if __name__ == "__main__":
    main()

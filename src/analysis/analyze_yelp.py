import json
import os
from collections import Counter

# Resolve project root from this script's location: src/analysis/ -> project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "yelp_academic_dataset_business.json")

def main():
    print("Reading dataset...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        businesses = json.load(f)

    total = len(businesses)
    print(f"Total businesses loaded: {total:,}\n")

    # ------------------------------------------------------------------ #
    # 1. Top 20 businesses by review_count
    # ------------------------------------------------------------------ #
    sorted_biz = sorted(businesses, key=lambda b: b.get("review_count", 0), reverse=True)
    top20 = sorted_biz[:20]

    print("=" * 130)
    print("TOP 20 BUSINESSES BY REVIEW COUNT")
    print("=" * 130)
    header = (
        f"{'#':>3}  {'Name':<40} {'City':<20} {'ST':<5} "
        f"{'Stars':>5} {'Reviews':>8} {'Open':>4}  {'Categories'}"
    )
    print(header)
    print("-" * 130)
    for i, b in enumerate(top20, 1):
        name = (b.get("name") or "")[:38]
        city = (b.get("city") or "")[:18]
        state = (b.get("state") or "")[:4]
        stars = b.get("stars", 0)
        reviews = b.get("review_count", 0)
        is_open = "Yes" if b.get("is_open", 0) == 1 else "No"
        cats = (b.get("categories") or "")[:70]
        print(f"{i:>3}  {name:<40} {city:<20} {state:<5} "
              f"{stars:>5.1f} {reviews:>8,} {is_open:>4}  {cats}")
    print()

    # ------------------------------------------------------------------ #
    # 2. Breakdown by state (top 10)
    # ------------------------------------------------------------------ #
    state_counter = Counter(b.get("state", "Unknown") for b in businesses)
    top_states = state_counter.most_common(10)

    print("=" * 60)
    print("TOP 10 STATES BY NUMBER OF BUSINESSES")
    print("=" * 60)
    print(f"{'#':>3}  {'State':<10} {'Count':>10} {'% of Total':>12}")
    print("-" * 40)
    for i, (state, count) in enumerate(top_states, 1):
        pct = count / total * 100
        print(f"{i:>3}  {state:<10} {count:>10,} {pct:>11.1f}%")
    print()

    # ------------------------------------------------------------------ #
    # 3. Top individual categories
    # ------------------------------------------------------------------ #
    cat_counter = Counter()
    for b in businesses:
        cats_str = b.get("categories") or ""
        if cats_str:
            for cat in cats_str.split(","):
                cat = cat.strip()
                if cat:
                    cat_counter[cat] += 1

    top_cats = cat_counter.most_common(25)

    print("=" * 60)
    print("TOP 25 INDIVIDUAL CATEGORIES")
    print("=" * 60)
    print(f"{'#':>3}  {'Category':<35} {'Count':>10} {'% of Biz':>10}")
    print("-" * 60)
    for i, (cat, count) in enumerate(top_cats, 1):
        pct = count / total * 100
        print(f"{i:>3}  {cat:<35} {count:>10,} {pct:>9.1f}%")
    print()

    # ------------------------------------------------------------------ #
    # 4. Quick summary stats
    # ------------------------------------------------------------------ #
    open_count = sum(1 for b in businesses if b.get("is_open", 0) == 1)
    avg_stars = sum(b.get("stars", 0) for b in businesses) / total if total else 0
    avg_reviews = sum(b.get("review_count", 0) for b in businesses) / total if total else 0

    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"  Total businesses:       {total:>10,}")
    print(f"  Currently open:         {open_count:>10,}  ({open_count/total*100:.1f}%)")
    print(f"  Currently closed:       {total - open_count:>10,}  ({(total-open_count)/total*100:.1f}%)")
    print(f"  Average star rating:    {avg_stars:>10.2f}")
    print(f"  Average review count:   {avg_reviews:>10.1f}")
    print(f"  Unique states/regions:  {len(state_counter):>10,}")
    print(f"  Unique categories:      {len(cat_counter):>10,}")
    print()

if __name__ == "__main__":
    main()

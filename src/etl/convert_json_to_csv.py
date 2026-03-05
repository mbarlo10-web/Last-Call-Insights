"""
Convert Yelp JSON dataset files to CSV format.
Handles JSON arrays and JSONL. Flattens nested dicts (attributes, hours) in business.json.
"""

import json
import os
import time
import pandas as pd

# Resolve project root from this script's location: src/etl/ -> project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

FILES = [
    ("yelp_academic_dataset_business.json", "business.csv"),
    ("yelp_academic_dataset_review.json",   "review.csv"),
    ("yelp_academic_dataset_user.json",     "user.csv"),
    ("yelp_academic_dataset_tip.json",      "tip.csv"),
    ("yelp_academic_dataset_checkin.json",   "checkin.csv"),
]

def load_json_flexible(filepath):
    """Try json.load (JSON array) first; fall back to JSONL line-by-line."""
    print(f"  Attempting json.load (JSON array)...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"  -> Loaded as JSON array: {len(data)} records")
            return data
    except (json.JSONDecodeError, MemoryError) as e:
        print(f"  -> json.load failed ({type(e).__name__}), falling back to JSONL...")

    # JSONL fallback
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Skip leading '[' or trailing ']' if it's a pretty-printed array saved one-per-line
            if line == "[" or line == "]":
                continue
            # Strip trailing comma from array-style JSONL
            if line.endswith(","):
                line = line[:-1]
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARNING: skipping malformed line {i+1}")
    print(f"  -> Loaded as JSONL: {len(records)} records")
    return records


def flatten_business_record(rec):
    """Flatten nested 'attributes' and 'hours' dicts into top-level keys."""
    flat = {}
    for key, val in rec.items():
        if key == "attributes" and isinstance(val, dict):
            for akey, aval in val.items():
                flat[f"attributes.{akey}"] = aval
        elif key == "hours" and isinstance(val, dict):
            for hkey, hval in val.items():
                flat[f"hours.{hkey}"] = hval
        else:
            flat[key] = val
    return flat


def convert_file(src_name, dst_name, is_business=False):
    src_path = os.path.join(SRC_DIR, src_name)
    dst_path = os.path.join(OUT_DIR, dst_name)

    src_size_mb = os.path.getsize(src_path) / (1024 * 1024)
    print(f"\n{'='*70}")
    print(f"Processing: {src_name} ({src_size_mb:.1f} MB)")
    print(f"{'='*70}")

    t0 = time.time()
    data = load_json_flexible(src_path)

    if is_business:
        print("  Flattening nested attributes/hours fields...")
        data = [flatten_business_record(r) for r in data]

    print("  Building DataFrame...")
    df = pd.DataFrame(data)

    # For any remaining columns that are dicts/lists, convert to string
    for col in df.columns:
        sample = df[col].dropna().head(5)
        if any(isinstance(v, (dict, list)) for v in sample):
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)

    print(f"  Writing CSV -> {dst_path}")
    df.to_csv(dst_path, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    dst_size_mb = os.path.getsize(dst_path) / (1024 * 1024)

    print(f"\n  RESULT: {dst_name}")
    print(f"    Rows:    {len(df):,}")
    print(f"    Columns: {len(df.columns)}")
    print(f"    Column names: {list(df.columns)}")
    print(f"    CSV size: {dst_size_mb:.1f} MB")
    print(f"    Time: {elapsed:.1f}s")
    return dst_name, len(df), len(df.columns), dst_size_mb


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    results = []
    for src_name, dst_name in FILES:
        is_business = "business" in src_name
        info = convert_file(src_name, dst_name, is_business=is_business)
        results.append(info)

    # Summary
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"{'File':<20} {'Rows':>12} {'Cols':>8} {'CSV Size (MB)':>15}")
    print("-" * 60)
    for name, rows, cols, size in results:
        print(f"{name:<20} {rows:>12,} {cols:>8} {size:>15.1f}")
    print("-" * 60)
    print("All files converted successfully!")


if __name__ == "__main__":
    main()

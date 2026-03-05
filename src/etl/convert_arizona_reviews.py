"""
Convert review-Arizona.json (JSONL, ~18M lines) to CSV.
Processes in chunks to avoid memory issues with the 5.7 GB file.
Flattens the nested 'resp' field into resp_time and resp_text columns.
"""

import json
import os
import time
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "review-Arizona.json")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "review_arizona.csv")

CHUNK_SIZE = 100_000  # rows per chunk


def flatten_record(rec):
    """Flatten nested 'resp' dict and convert 'pics' list to count."""
    flat = {}
    for key, val in rec.items():
        if key == "resp" and isinstance(val, dict):
            flat["resp_time"] = val.get("time")
            flat["resp_text"] = val.get("text")
        elif key == "pics" and isinstance(val, list):
            flat["pics_count"] = len(val)
        elif key == "pics" and val is None:
            flat["pics_count"] = 0
        else:
            flat[key] = val
    return flat


def main():
    print(f"Source: {SRC_PATH}")
    print(f"Output: {OUT_PATH}")
    src_size_mb = os.path.getsize(SRC_PATH) / (1024 * 1024)
    print(f"File size: {src_size_mb:,.1f} MB")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    t0 = time.time()
    total_rows = 0
    chunk = []
    first_chunk = True
    skipped = 0

    with open(SRC_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                chunk.append(flatten_record(rec))
            except json.JSONDecodeError:
                skipped += 1
                continue

            if len(chunk) >= CHUNK_SIZE:
                df = pd.DataFrame(chunk)
                df.to_csv(OUT_PATH, index=False, encoding="utf-8",
                          mode="w" if first_chunk else "a",
                          header=first_chunk)
                total_rows += len(df)
                elapsed = time.time() - t0
                print(f"  Written {total_rows:>12,} rows  |  {elapsed:.0f}s elapsed")
                chunk = []
                first_chunk = False

    # Write remaining rows
    if chunk:
        df = pd.DataFrame(chunk)
        df.to_csv(OUT_PATH, index=False, encoding="utf-8",
                  mode="w" if first_chunk else "a",
                  header=first_chunk)
        total_rows += len(df)

    elapsed = time.time() - t0
    out_size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total rows:     {total_rows:>12,}")
    print(f"  Skipped lines:  {skipped:>12,}")
    print(f"  CSV size:       {out_size_mb:>12,.1f} MB")
    print(f"  Time:           {elapsed:>12,.1f}s")


if __name__ == "__main__":
    main()

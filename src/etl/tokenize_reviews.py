"""
Sentence Segmentation + Tokenization of Tucson nightlife reviews.
Uses NLTK to process the 'text' column and adds:
  - sentence_count: number of sentences
  - token_count: number of meaningful tokens (no punct/spaces)
  - sentences: pipe-delimited sentence strings
  - tokens: pipe-delimited lowercased tokens (no punct/spaces)
"""

import os
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")


def process_review(text):
    """Segment into sentences and tokenize a single review."""
    sentences = sent_tokenize(text)
    tokens = [t.lower() for t in word_tokenize(text) if t.isalnum()]
    return {
        "sentence_count": len(sentences),
        "token_count": len(tokens),
        "sentences": "|".join(s.strip() for s in sentences),
        "tokens": "|".join(tokens),
    }


def main():
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    print("Processing reviews with NLTK...")
    results = df["text"].fillna("").apply(process_review)
    result_df = pd.DataFrame(results.tolist())

    df["sentence_count"] = result_df["sentence_count"]
    df["token_count"] = result_df["token_count"]
    df["sentences"] = result_df["sentences"]
    df["tokens"] = result_df["tokens"]

    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"\nSaved to: {CSV_PATH}")
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    print(f"\n{'='*50}")
    print("TOKENIZATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Avg sentences/review:  {df['sentence_count'].mean():.1f}")
    print(f"  Avg tokens/review:     {df['token_count'].mean():.1f}")
    print(f"  Min sentences:         {df['sentence_count'].min()}")
    print(f"  Max sentences:         {df['sentence_count'].max()}")
    print(f"  Min tokens:            {df['token_count'].min()}")
    print(f"  Max tokens:            {df['token_count'].max()}")


if __name__ == "__main__":
    main()

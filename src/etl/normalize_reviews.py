"""
Normalization + Lemmatization of tokenized Tucson nightlife reviews.
Uses NLTK WordNetLemmatizer with POS tagging for accurate lemmatization.
Removes punctuation and stop words.
Adds to business_reviews_tucson.csv:
  - tokens_normalized: pipe-delimited lowercased, lemmatized tokens (no punct/stopwords)
"""

import os
import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

STOP_WORDS = set(stopwords.words("english"))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")


def get_wordnet_pos(tag):
    """Map POS tag to WordNet POS for lemmatizer."""
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def normalize_and_lemmatize(text):
    """Lowercase, tokenize, POS-tag, and lemmatize a review."""
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalnum() and w not in STOP_WORDS]
    tagged = pos_tag(words)
    lemmas = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]
    return "|".join(lemmas)


def main():
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    print("Normalizing and lemmatizing reviews...")
    df["tokens_normalized"] = df["text"].fillna("").apply(normalize_and_lemmatize)

    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"\nSaved to: {CSV_PATH}")
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # Show a sample
    sample = df[["text", "tokens", "tokens_normalized"]].iloc[0]
    print(f"\n{'='*50}")
    print("SAMPLE (first review)")
    print(f"{'='*50}")
    print(f"Original tokens:    {sample['tokens'][:120]}...")
    print(f"Normalized/Lemma:   {sample['tokens_normalized'][:120]}...")


if __name__ == "__main__":
    main()

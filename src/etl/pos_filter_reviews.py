"""
POS Tagging + Noun/Adjective filtering of Tucson nightlife reviews.
Keeps only nouns (NN, NNS, NNP, NNPS) and adjectives (JJ, JJR, JJS)
from the normalized tokens.
Adds to business_reviews_tucson.csv:
  - pos_tags: pipe-delimited word_TAG pairs
  - tokens_nouns_adj: pipe-delimited tokens filtered to nouns & adjectives only
"""

import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

STOP_WORDS = set(stopwords.words("english"))
KEEP_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "business_reviews_tucson.csv")


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def tag_and_filter(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalnum() and w not in STOP_WORDS]
    tagged = pos_tag(words)
    lemmas_tagged = [(lemmatizer.lemmatize(w, get_wordnet_pos(t)), t) for w, t in tagged]

    pos_str = "|".join(f"{w}_{t}" for w, t in lemmas_tagged)
    filtered = "|".join(w for w, t in lemmas_tagged if t in KEEP_TAGS)
    return pos_str, filtered


def main():
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    print("POS tagging and filtering to nouns & adjectives...")
    results = df["text"].fillna("").apply(tag_and_filter)
    df["pos_tags"] = results.apply(lambda x: x[0])
    df["tokens_nouns_adj"] = results.apply(lambda x: x[1])

    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"\nSaved to: {CSV_PATH}")
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # Sample
    sample = df.iloc[0]
    print(f"\n{'='*60}")
    print("SAMPLE (first review)")
    print(f"{'='*60}")
    print(f"POS tags:       {sample['pos_tags'][:150]}...")
    print(f"Nouns & Adj:    {sample['tokens_nouns_adj'][:150]}...")

    # Stats
    noun_adj_counts = df["tokens_nouns_adj"].apply(lambda x: len(x.split("|")) if x else 0)
    print(f"\n{'='*60}")
    print("NOUN & ADJECTIVE FILTER STATS")
    print(f"{'='*60}")
    print(f"  Avg nouns+adj per review:  {noun_adj_counts.mean():.1f}")
    print(f"  Min:                       {noun_adj_counts.min()}")
    print(f"  Max:                       {noun_adj_counts.max()}")


if __name__ == "__main__":
    main()

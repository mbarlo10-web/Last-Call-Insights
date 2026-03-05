# Tucson Nightlife Sentiment Analysis
### Last Call Insights — NL Consulting

> *"Stars don't tell the full story."*

NLP-driven sentiment analysis of **1,226 Yelp reviews** across **39 Tucson nightlife venues** (2005–2018). This project moves beyond star ratings to surface emotional volatility, dissatisfaction hotspots, and the gap between perceived and actual customer sentiment.

---

## Business Question

> *Do star ratings accurately reflect emotional sentiment and service consistency in nightlife venues?*

---

## Methods

| Method | Purpose |
|---|---|
| **VADER Sentiment Analysis** | Score each review from −1 (negative) to +1 (positive) |
| **Aspect-Based Sentiment (ABSA)** | Tag reviews by dimension: Food, Service, Atmosphere, Value, Location |
| **LDA Topic Modeling** | Unsupervised discovery of 21 customer conversation clusters |
| **Unhappiness Index** | Identify venues generating disproportionate negative feedback |
| **Rating–Sentiment Gap** | Expose venues overrated or underrated by star scores |

---

## Key Findings

- **0.31** average sentiment score — overall positive, but wide variance masks risk
- **21 topics** discovered via LDA — dining, bar scene, tacos, service quality, and more
- **Prep & Pastry** leads the Unhappiness Index at 18.3 (79/341 reviews below threshold)
- Star ratings and sentiment scores **do not align** across multiple venues

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   └── category engineering.txt
│   └── processed/
│       └── business_reviews_tucson.csv   # Main dataset (1,226 reviews, 49 columns)
├── src/
│   ├── etl/                              # Data pipeline
│   │   ├── convert_json_to_csv.py        # Raw Yelp JSON → CSV
│   │   ├── convert_arizona_reviews.py    # Filter to Tucson nightlife
│   │   ├── tokenize_reviews.py           # Sentence segmentation + tokenization
│   │   ├── normalize_reviews.py          # Lemmatization + stopword removal
│   │   └── pos_filter_reviews.py         # POS tagging + noun/adjective extraction
│   └── analysis/                         # Analysis scripts
│       ├── data_quality_check.py         # Missing values, duplicates, range validation
│       ├── sentiment_by_aspect.py        # ABSA sentiment breakdown
│       ├── sentiment_trend.py            # Quarterly sentiment trend over time
│       ├── sentiment_trend_venues.py     # Per-venue sentiment slope (improving/declining)
│       ├── topic_modeling.py             # LDA topic modeling
│       ├── topic_visualizations.py       # Topic charts and word distributions
│       ├── unhappiness_index.py          # Unhappiness Index by venue
│       ├── rating_sentiment_gap.py       # Star rating vs sentiment gap
│       ├── top_nouns_adjectives.py       # Most frequent words in reviews
│       ├── per_restaurant_analysis.py    # Individual venue deep dives
│       └── engineered_category_frequency.py  # Venue type and food emphasis charts
├── Images/
│   ├── preliminary findings/             # Core analysis visualizations
│   ├── slide_charts/                     # Dark-themed charts for presentation
│   ├── statistics/                       # Dataset summary charts
│   ├── Topic Modeling/                   # LDA visualizations
│   └── per_restaurant/                   # Per-venue charts
└── requirements.txt
```

---

## Dataset

| Attribute | Value |
|---|---|
| Source | Yelp Academic Dataset |
| Venues | 39 Tucson nightlife businesses |
| Reviews | 1,226 |
| Time span | March 2005 – October 2018 |
| Avg sentiment score | 0.31 |
| Avg review rating | 4.05 ★ |
| Vocabulary size | 5,723 unique words (normalized) |

> **Note:** Raw Yelp JSON files are excluded from this repo due to size. Only the processed dataset (`business_reviews_tucson.csv`) is included.

---

## Data Quality

- Zero nulls in all core analysis columns
- Zero exact duplicate reviews
- 19 `attributes.*` columns have high missingness (18–91%) — excluded from analysis
- 92 same-day reviews from the same venue with different text — confirmed as legitimate

---

## Setup

```bash
pip install -r requirements.txt
```

Run the data quality check:
```bash
python src/analysis/data_quality_check.py
```

---

## Course

**CIS 509 — Unstructured Data Analytics**
W. P. Carey School of Business, Arizona State University

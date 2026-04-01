#!/usr/bin/env python3
"""Select 30 high-quality Business probes from AG News test.csv (first 3000 rows).

Output: data/ag_news_business_30.json

Selection criteria:
  - Label "3" in CSV (Business in AG News 1-based scheme -> 2 in 0-based)
  - news_text = title + " " + text  (matches data_loader.py)
  - Clean text (no HTML tags, decoded entities)
  - Word count between 40 and 200
  - Topic diversity across business sub-domains
"""

import csv
import json
import re
import sys
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "AG_News_Datasets" / "test.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ag_news_business_30.json"
MAX_ROWS = 3000

BUSINESS_KEYWORDS = {
    "oil_energy": ["oil", "crude", "barrel", "opec", "pipeline", "refinery", "energy", "fuel", "gasoline", "petroleum"],
    "stock_ipo": ["stock", "shares", "ipo", "nasdaq", "dow", "s&p", "index", "rally", "wall street", "investor"],
    "retail": ["retail", "store", "sales", "consumer", "shopping", "walmart", "kmart"],
    "airline": ["airline", "flight", "airbus", "boeing", "aviation", "carrier", "airport"],
    "merger_finance": ["merger", "acquisition", "buyout", "takeover", "bid", "deal", "bank"],
    "pharma": ["drug", "pharmaceutical", "biotech", "medicine", "fda", "vaccine"],
    "tech_biz": ["samsung", "intel", "chip", "semiconductor", "nortel", "telecom"],
    "layoff": ["layoff", "cut", "workforce", "restructuring", "job", "pension"],
    "macro": ["gdp", "inflation", "unemployment", "economic", "growth", "recession", "deficit", "trade"],
    "regulation": ["sec", "regulation", "lawsuit", "antitrust", "court", "fine", "probe"],
}

def clean_text(raw: str) -> str:
    t = raw.strip()
    t = re.sub(r"&lt;.*?&gt;", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace(" #39;", "'").replace("#39;", "'")
    t = t.replace(" amp;", " &").replace("&amp;", "&")
    t = t.replace("&lt;", "<").replace("&gt;", ">")
    t = t.replace("&quot;", '"')
    t = t.replace(" #36;", "$").replace("#36;", "$")
    t = t.replace("\\", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def word_count(text: str) -> int:
    return len(text.split())


def has_html_junk(text: str) -> bool:
    return bool(re.search(r'HREF=|<A |<B>|<b>|&lt;A |target=', text, re.IGNORECASE))


def classify_topic(text: str) -> str:
    tl = text.lower()
    scores = {}
    for topic, kws in BUSINESS_KEYWORDS.items():
        scores[topic] = sum(1 for kw in kws if kw in tl)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def main():
    if not CSV_PATH.is_file():
        print(f"CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    candidates = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader, start=1):
            if row_idx > MAX_ROWS:
                break
            if len(row) < 3:
                continue
            label, title, text = row[0], row[1], row[2]
            if label.strip() != "3":
                continue
            news_text = clean_text(title) + " " + clean_text(text)
            if has_html_junk(news_text):
                continue
            wc = word_count(news_text)
            if wc < 40 or wc > 200:
                continue
            topic = classify_topic(news_text)
            candidates.append({
                "csv_row": row_idx,
                "news_text": news_text,
                "word_count": wc,
                "topic": topic,
            })

    print(f"Candidates after filtering: {len(candidates)}")
    topic_counts = {}
    for c in candidates:
        topic_counts[c["topic"]] = topic_counts.get(c["topic"], 0) + 1
    print("Topic distribution:")
    for t, cnt in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {cnt}")

    # Greedy selection: pick from each topic, then fill remaining
    selected = []
    by_topic = {}
    for c in candidates:
        by_topic.setdefault(c["topic"], []).append(c)

    target_per_topic = {
        "oil_energy": 4,
        "stock_ipo": 3,
        "retail": 3,
        "airline": 3,
        "merger_finance": 4,
        "pharma": 2,
        "tech_biz": 3,
        "layoff": 3,
        "macro": 3,
        "regulation": 2,
    }

    used_rows = set()
    for topic, target_n in target_per_topic.items():
        pool = by_topic.get(topic, [])
        # prefer mid-length articles (60-120 words)
        pool.sort(key=lambda c: abs(c["word_count"] - 90))
        for c in pool:
            if len(selected) >= 30:
                break
            if c["csv_row"] not in used_rows and sum(1 for s in selected if s["topic"] == topic) < target_n:
                selected.append(c)
                used_rows.add(c["csv_row"])

    # Fill from "general" or any remaining topic if needed
    if len(selected) < 30:
        remaining = [c for c in candidates if c["csv_row"] not in used_rows]
        remaining.sort(key=lambda c: abs(c["word_count"] - 90))
        for c in remaining:
            if len(selected) >= 30:
                break
            selected.append(c)
            used_rows.add(c["csv_row"])

    # Sort by CSV row order for reproducibility
    selected.sort(key=lambda c: c["csv_row"])

    print(f"\nSelected {len(selected)} samples:")
    final_topic_counts = {}
    for s in selected:
        final_topic_counts[s["topic"]] = final_topic_counts.get(s["topic"], 0) + 1
    for t, cnt in sorted(final_topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {cnt}")

    # Write JSON
    probes = []
    for idx, s in enumerate(selected, start=1):
        probes.append({
            "id": idx,
            "dataset_label_id": 2,
            "dataset_category": "Business",
            "news_text": s["news_text"],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(probes, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(probes)} probes to {OUT_PATH}")

    # Print summary
    for p in probes:
        preview = p["news_text"][:80] + ("..." if len(p["news_text"]) > 80 else "")
        print(f"  [{p['id']:2d}] {preview}")


if __name__ == "__main__":
    main()

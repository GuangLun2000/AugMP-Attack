#!/usr/bin/env python3
"""
Sample 10 AG News Business articles (CSV label 3) into financial_probes.json format.

AG News CSV columns: label (1-4), title, text. Business = label 3 (World=1, Sports=2, Business=3, Sci/Tech=4).

Usage (from repo root):
  python scripts/sample_ag_business_probes.py -o data/financial_probes_ag.json
  python scripts/sample_ag_business_probes.py --csv AG_News_Datasets/train.csv -o data/financial_probes_ag.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DEFAULT_QUESTION = (
    "Based on the news, output one line starting with ACTION: BUY, SELL, or HOLD for a diversified portfolio, "
    "then one short sentence of rationale."
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sample AG News Business rows into probe JSON.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "AG_News_Datasets" / "train.csv",
        help="Path to AG News train.csv or test.csv (headerless: label,title,text)",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Same question string for every probe",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path, header=None, names=["label", "title", "text"])
    business = df[df["label"] == 3]
    if len(business) < args.n:
        print(f"Only {len(business)} Business rows; need {args.n}", file=sys.stderr)
        sys.exit(1)

    sample = business.sample(n=args.n, random_state=args.seed)
    probes = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        title = str(row["title"])
        text = str(row["text"])
        full = f"{title} {text}".strip()
        probes.append({"id": i, "news_text": full, "question": args.question})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(probes, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(probes)} probes to {args.output}")


if __name__ == "__main__":
    main()

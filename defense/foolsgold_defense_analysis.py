#!/usr/bin/env python3
"""
FoolsGold-Inspired Defense Analysis for Federated Learning

This module analyzes per-round, per-client cosine similarity tables and computes
FoolsGold-style trust weights plus attacker detection metrics.

Important note:
Standard FoolsGold operates on historical client update vectors and their full
pairwise cosine similarity matrix. The table embedded below only provides one
scalar per client per round, so this script implements a proxy version:

  1. Treat each client's cosine value as its pairwise-mean similarity signal
     for that round (best match when server similarity mode is "pairwise")
  2. Accumulate the signal over rounds to form a historical similarity profile
  3. Down-weight clients whose historical similarity stays consistently high

This is therefore a FoolsGold-inspired detector, not an exact reproduction of
the original paper when a full similarity matrix is unavailable.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Data Parsing
# =============================================================================

def parse_metric_table(text: str) -> Tuple[Dict[int, Dict[int, float]], List[int]]:
    """
    Parse a metric table in the format:
        Round | Client0(B) | Client1(B) | ... | Client6(A) | Mean | Std

    Returns:
        data: {round_num: {client_id: value}}
        client_ids: [0, 1, 2, ...]
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return {}, []

    header = lines[0]
    client_pattern = re.compile(r"Client(\d+)\s*\([BA]\)")
    client_ids = [int(match) for match in client_pattern.findall(header)]
    num_clients = len(client_ids)

    data: Dict[int, Dict[int, float]] = {}
    for line in lines[1:]:
        if line.startswith("-"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < num_clients + 2:
            continue
        try:
            round_num = int(parts[0])
        except ValueError:
            continue

        row: Dict[int, float] = {}
        for idx, cid in enumerate(client_ids, start=1):
            try:
                row[cid] = float(parts[idx])
            except (ValueError, IndexError):
                pass
        if row:
            data[round_num] = row

    return data, client_ids


def parse_attacker_ids_from_header(header_line: str) -> List[int]:
    """Extract attacker IDs from header labels '(A)'."""
    client_pattern = re.compile(r"Client(\d+)\s*\(([BA])\)")
    attackers: List[int] = []
    for match in client_pattern.finditer(header_line):
        cid, label = int(match.group(1)), match.group(2)
        if label == "A":
            attackers.append(cid)
    return attackers


# =============================================================================
# FoolsGold Proxy
# =============================================================================

def compute_foolsgold_proxy_weights(
    historical_similarity: Dict[int, float],
    client_ids: List[int],
    apply_logit: bool = True,
) -> Dict[int, float]:
    """
    Convert historical similarity scores into FoolsGold-style trust weights.

    Higher historical similarity -> more suspicious -> lower weight.
    """
    if not client_ids:
        return {}

    values = np.array([historical_similarity.get(cid, 0.0) for cid in client_ids], dtype=float)
    min_val = float(values.min())
    max_val = float(values.max())

    if np.isclose(max_val, min_val):
        return {cid: 1.0 for cid in client_ids}

    suspiciousness = (values - min_val) / (max_val - min_val)
    weights = 1.0 - suspiciousness

    if apply_logit:
        weights = weights / max(weights.max(), 1e-12)
        weights = np.clip(weights, 1e-6, 1.0 - 1e-6)
        weights = np.log(weights / (1.0 - weights)) + 0.5
        weights = np.clip(weights, 0.0, 1.0)

    return {cid: float(weight) for cid, weight in zip(client_ids, weights)}


def detect_attackers_foolsgold_proxy(
    historical_similarity: Dict[int, float],
    client_ids: List[int],
    num_byzantine: int,
    apply_logit: bool = True,
) -> Tuple[List[int], Dict[int, float]]:
    """
    Detect attackers by selecting clients with the smallest FoolsGold-style weights.
    """
    weights = compute_foolsgold_proxy_weights(
        historical_similarity=historical_similarity,
        client_ids=client_ids,
        apply_logit=apply_logit,
    )

    if num_byzantine <= 0:
        return [], weights

    ranked = sorted(
        client_ids,
        key=lambda cid: (weights.get(cid, 1.0), -historical_similarity.get(cid, 0.0), cid),
    )
    return ranked[:num_byzantine], weights


# =============================================================================
# Detection Rate Computation
# =============================================================================

@dataclass
class DetectionResult:
    """Result of FoolsGold proxy analysis for one round."""

    round_num: int
    historical_similarity: Dict[int, float]
    weights: Dict[int, float]
    detected_attackers: List[int]
    true_attackers: List[int]
    benign_clients: List[int]
    false_positives: List[int]
    true_positives: List[int]
    false_negatives: List[int]


def analyze_round(
    round_num: int,
    historical_similarity: Dict[int, float],
    client_ids: List[int],
    attacker_ids: List[int],
    num_byzantine: int,
    apply_logit: bool = True,
) -> DetectionResult:
    """Run proxy FoolsGold detection for one round."""
    benign_ids = [cid for cid in client_ids if cid not in attacker_ids]
    detected, weights = detect_attackers_foolsgold_proxy(
        historical_similarity=historical_similarity,
        client_ids=client_ids,
        num_byzantine=num_byzantine,
        apply_logit=apply_logit,
    )

    true_positives = [cid for cid in detected if cid in attacker_ids]
    false_positives = [cid for cid in detected if cid in benign_ids]
    false_negatives = [cid for cid in attacker_ids if cid not in detected]

    return DetectionResult(
        round_num=round_num,
        historical_similarity=dict(historical_similarity),
        weights=weights,
        detected_attackers=detected,
        true_attackers=list(attacker_ids),
        benign_clients=benign_ids,
        false_positives=false_positives,
        true_positives=true_positives,
        false_negatives=false_negatives,
    )


def compute_detection_rates(results: List[DetectionResult]) -> Dict[str, float]:
    """Compute aggregate detection metrics."""
    if not results:
        return {}

    total_rounds = len(results)
    total_attacker_rounds = sum(len(result.true_attackers) for result in results)
    total_tp = sum(len(result.true_positives) for result in results)
    total_fp = sum(len(result.false_positives) for result in results)
    total_fn = sum(len(result.false_negatives) for result in results)

    if total_attacker_rounds > 0:
        rounds_with_detection = sum(1 for result in results if result.true_positives)
        attack_presence_detection_rate = rounds_with_detection / total_rounds
    else:
        rounds_with_detection = sum(1 for result in results if result.detected_attackers)
        attack_presence_detection_rate = rounds_with_detection / total_rounds

    attacker_identification_rate = (
        total_tp / total_attacker_rounds if total_attacker_rounds else 0.0
    )
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "attack_presence_detection_rate": attack_presence_detection_rate,
        "attacker_identification_rate": attacker_identification_rate,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_rounds": total_rounds,
        "rounds_with_detection": rounds_with_detection,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
    }


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_foolsgold_analysis(
    cosine_text: str,
    attacker_ids: Optional[List[int]] = None,
    history_mode: str = "mean",
    apply_logit: bool = True,
) -> Dict:
    """
    Run FoolsGold-style analysis on a cosine similarity table.

    Args:
        cosine_text: Raw cosine similarity table
        attacker_ids: Optional attacker IDs; inferred from header if omitted
        history_mode: "mean" or "sum" for the historical similarity profile
        apply_logit: Whether to apply a FoolsGold-style logit transform
    """
    sim_data, client_ids = parse_metric_table(cosine_text)
    if not sim_data:
        return {"error": "Failed to parse cosine similarity data"}

    if attacker_ids is None:
        header = cosine_text.strip().split("\n")[0]
        attacker_ids = parse_attacker_ids_from_header(header)

    rounds = sorted(sim_data.keys())
    running_sum = {cid: 0.0 for cid in client_ids}
    results: List[DetectionResult] = []

    for idx, round_num in enumerate(rounds, start=1):
        round_values = sim_data[round_num]
        for cid in client_ids:
            running_sum[cid] += round_values.get(cid, 0.0)

        if history_mode == "sum":
            historical_similarity = dict(running_sum)
        else:
            historical_similarity = {
                cid: running_sum[cid] / idx for cid in client_ids
            }

        result = analyze_round(
            round_num=round_num,
            historical_similarity=historical_similarity,
            client_ids=client_ids,
            attacker_ids=attacker_ids,
            num_byzantine=len(attacker_ids),
            apply_logit=apply_logit,
        )
        results.append(result)

    return {
        "client_ids": client_ids,
        "attacker_ids": attacker_ids,
        "rounds": rounds,
        "history_mode": history_mode,
        "apply_logit": apply_logit,
        "per_round_results": results,
        "detection_rates": compute_detection_rates(results),
    }


def print_analysis_report(analysis: Dict) -> None:
    """Print a human-readable analysis report."""
    print("\n" + "=" * 80)
    print("FOOLSGOLD-INSPIRED DEFENSE ANALYSIS REPORT")
    print("=" * 80)
    print(f"Clients: {analysis['client_ids']}")
    print(f"Attackers (ground truth): {analysis['attacker_ids']}")
    print(f"Rounds: {len(analysis['rounds'])}")
    print(f"History mode: {analysis['history_mode']}")
    print(f"Logit transform: {analysis['apply_logit']}")
    print("=" * 80)

    rates = analysis["detection_rates"]
    print(f"Attack Presence Detection Rate: {rates['attack_presence_detection_rate']:.2%}")
    print(f"Attacker Identification Rate:   {rates['attacker_identification_rate']:.2%}")
    print(
        f"Precision: {rates['precision']:.2%}  "
        f"Recall: {rates['recall']:.2%}  "
        f"F1: {rates['f1_score']:.2%}"
    )
    print(
        f"TP={rates['total_true_positives']} "
        f"FP={rates['total_false_positives']} "
        f"FN={rates['total_false_negatives']}"
    )

    if analysis["per_round_results"]:
        final_round = analysis["per_round_results"][-1]
        ordered = sorted(
            analysis["client_ids"],
            key=lambda cid: final_round.weights.get(cid, 1.0),
        )
        print("\nFinal-round historical similarity and weights:")
        for cid in ordered:
            hist = final_round.historical_similarity.get(cid, 0.0)
            weight = final_round.weights.get(cid, 0.0)
            label = "A" if cid in analysis["attacker_ids"] else "B"
            print(
                f"  Client{cid}({label}): "
                f"historical_similarity={hist:.6f}, weight={weight:.6f}"
            )


# =============================================================================
# CLI with Embedded Data
# =============================================================================

EMBEDDED_COSINE_TABLE = """
Round | Client0(B) | Client1(B) | Client2(B) | Client3(B) | Client4(B) | Client5(A) | Client6(A) | Mean | Std
--------------------------------------------------------------------------------
1      | 0.450464       | 0.431138       | 0.415028       | 0.536548       | 0.533065       | 0.416836       | 0.439088       | 0.460310 | 0.048468
2      | 0.557362       | 0.550317       | 0.544497       | 0.580768       | 0.574892       | 0.589824       | 0.535950       | 0.561944 | 0.018653
3      | 0.634930       | 0.658320       | 0.648879       | 0.630076       | 0.660738       | 0.639013       | 0.568094       | 0.634293 | 0.029062
4      | 0.411259       | 0.428126       | 0.420907       | 0.393263       | 0.428657       | 0.386418       | 0.370841       | 0.405639 | 0.020818
5      | 0.209074       | 0.182490       | 0.178149       | 0.179688       | 0.229369       | 0.196596       | 0.157848       | 0.190459 | 0.021688
6      | 0.139681       | 0.095974       | 0.093781       | 0.126241       | 0.161644       | 0.192253       | 0.091754       | 0.128761 | 0.035669
7      | 0.136514       | 0.052176       | 0.046776       | 0.136008       | 0.169697       | 0.147540       | 0.073988       | 0.108957 | 0.046264
8      | 0.253000       | 0.262755       | 0.245765       | 0.233601       | 0.272988       | 0.201522       | 0.181902       | 0.235933 | 0.030681
9      | 0.105185       | 0.024534       | 0.013258       | 0.102256       | 0.140701       | 0.107399       | 0.054425       | 0.078251 | 0.044286
10     | 0.127078       | 0.058230       | 0.055810       | 0.102980       | 0.155495       | 0.048674       | 0.051745       | 0.085716 | 0.039739
11     | 0.104135       | 0.000673       | -0.000901      | 0.080631       | 0.133210       | 0.081333       | 0.001784       | 0.057267 | 0.051760
12     | 0.080283       | 0.034781       | 0.036356       | 0.047691       | 0.098341       | 0.108618       | 0.025314       | 0.061626 | 0.031113
13     | 0.096992       | 0.019592       | 0.029567       | 0.070666       | 0.127449       | 0.095590       | 0.045732       | 0.069370 | 0.036735
14     | 0.078518       | -0.002617      | -0.001354      | 0.068391       | 0.094595       | 0.056651       | 0.015213       | 0.044200 | 0.036965
15     | 0.085750       | -0.006197      | 0.006777       | 0.070454       | 0.112701       | 0.104009       | 0.038539       | 0.058862 | 0.043347
16     | 0.064098       | 0.017298       | 0.022442       | 0.045115       | 0.071431       | 0.099410       | 0.035381       | 0.050739 | 0.027156
17     | 0.078127       | 0.053027       | 0.056453       | 0.059802       | 0.089529       | 0.094461       | 0.020453       | 0.064550 | 0.023508
18     | 0.074905       | -0.008097      | -0.009913      | 0.064203       | 0.080491       | 0.062781       | 0.037657       | 0.043147 | 0.035261
19     | 0.316690       | 0.346225       | 0.327050       | 0.331886       | 0.346682       | 0.271036       | 0.291665       | 0.318748 | 0.026161
20     | 0.133977       | 0.069007       | 0.072936       | 0.127657       | 0.159349       | 0.132909       | 0.074285       | 0.110017 | 0.034177
21     | 0.067513       | 0.039570       | 0.048470       | 0.036438       | 0.079834       | 0.084069       | 0.026809       | 0.054672 | 0.020831
22     | 0.056986       | 0.017194       | 0.020638       | 0.064807       | 0.063925       | 0.118283       | 0.020553       | 0.051770 | 0.033611
23     | 0.061784       | 0.020268       | 0.029512       | 0.061091       | 0.092623       | 0.105241       | 0.018623       | 0.055592 | 0.032071
24     | 0.071280       | 0.015425       | 0.025671       | 0.077080       | 0.080489       | 0.003422       | 0.025806       | 0.042739 | 0.029973
25     | 0.023238       | -0.019083      | -0.023408      | 0.024851       | 0.037454       | 0.121011       | 0.001507       | 0.023653 | 0.045043
26     | 0.195506       | 0.168237       | 0.174702       | 0.188044       | 0.206493       | 0.235619       | 0.100595       | 0.181314 | 0.038840
27     | 0.049773       | -0.004253      | -0.004546      | 0.075701       | 0.079639       | 0.060566       | 0.027716       | 0.040657 | 0.032634
28     | 0.121192       | 0.033589       | 0.061061       | 0.116475       | 0.129890       | 0.091926       | 0.028317       | 0.083207 | 0.039225
29     | 0.087317       | 0.035172       | 0.044316       | 0.047685       | 0.084823       | 0.098192       | 0.015399       | 0.058986 | 0.028828
30     | 0.040038       | -0.050990      | -0.037121      | 0.011609       | 0.074351       | 0.097153       | 0.011690       | 0.020961 | 0.050347
31     | 0.304049       | 0.353805       | 0.331880       | 0.320330       | 0.325942       | 0.212391       | 0.249562       | 0.299708 | 0.046649
32     | 0.081507       | 0.055785       | 0.059403       | 0.100991       | 0.108567       | 0.120400       | 0.002687       | 0.075620 | 0.037311
33     | 0.068547       | -0.007444      | 0.002820       | 0.054809       | 0.085212       | 0.125728       | 0.012795       | 0.048924 | 0.045109
34     | 0.073103       | 0.033498       | 0.046227       | 0.052599       | 0.086505       | 0.120887       | 0.034994       | 0.063973 | 0.029382
35     | 0.072129       | -0.034696      | -0.030783      | 0.076817       | 0.109479       | 0.128713       | 0.042585       | 0.052035 | 0.059363
36     | 0.130315       | 0.102040       | 0.118160       | 0.097463       | 0.135815       | 0.082148       | 0.031345       | 0.099612 | 0.032892
37     | 0.029182       | -0.047127      | -0.043112      | 0.008405       | 0.100180       | 0.114310       | 0.003934       | 0.023682 | 0.058791
38     | 0.151757       | 0.098581       | 0.113693       | 0.121152       | 0.149300       | 0.112135       | 0.043590       | 0.112887 | 0.033622
39     | 0.048828       | -0.024157      | -0.014526      | 0.074229       | 0.073050       | 0.078897       | 0.002296       | 0.034088 | 0.041612
40     | 0.033324       | -0.038762      | -0.050537      | 0.083376       | 0.057286       | 0.063416       | 0.001470       | 0.021368 | 0.048148
41     | 0.182136       | 0.155725       | 0.186605       | 0.128150       | 0.198476       | 0.209968       | 0.135553       | 0.170945 | 0.029196
42     | 0.038935       | -0.021860      | -0.020599      | 0.036064       | 0.062640       | 0.109717       | 0.008922       | 0.030546 | 0.043472
43     | 0.134393       | 0.066509       | 0.079290       | 0.116031       | 0.153646       | 0.074362       | 0.077685       | 0.100274 | 0.031675
44     | 0.064157       | -0.046562      | -0.038539      | 0.121296       | 0.075709       | 0.082572       | 0.014441       | 0.039011 | 0.059267
45     | 0.099790       | 0.066163       | 0.072253       | 0.064437       | 0.113945       | 0.100119       | 0.033293       | 0.078571 | 0.025661
46     | 0.047805       | -0.048026      | -0.042773      | 0.086318       | 0.049894       | 0.074753       | 0.004037       | 0.024572 | 0.050341
47     | 0.193501       | 0.175798       | 0.201316       | 0.149232       | 0.191596       | 0.167854       | 0.096171       | 0.167924 | 0.033572
48     | 0.031965       | -0.056213      | -0.053462      | 0.048863       | 0.037911       | 0.159470       | -0.035811      | 0.018961 | 0.070772
49     | 0.280704       | 0.319450       | 0.304150       | 0.263144       | 0.299604       | 0.175250       | 0.087077       | 0.247054 | 0.078742
50     | 0.067802       | 0.054973       | 0.053851       | 0.038550       | 0.097381       | 0.022450       | 0.044328       | 0.054191 | 0.022041
"""


def main() -> None:
    """
    Run FoolsGold-inspired analysis on embedded data or an external cosine table.

    Usage:
        python defense/foolsgold_defense_analysis.py
        python defense/foolsgold_defense_analysis.py cosine_table.txt
    """
    if len(sys.argv) > 1:
        cosine_path = Path(sys.argv[1])
        cosine_text = cosine_path.read_text(encoding="utf-8")
        print(f"Running FoolsGold-inspired analysis on: {cosine_path}")
    else:
        cosine_text = EMBEDDED_COSINE_TABLE
        print("Running FoolsGold-inspired analysis on embedded cosine table...")

    analysis = run_foolsgold_analysis(
        cosine_text=cosine_text,
        attacker_ids=None,
        history_mode="mean",
        apply_logit=True,
    )
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    print_analysis_report(analysis)
    print("\n" + "=" * 80)
    print("Note: Standard FoolsGold needs full pairwise historical similarities.")
    print("This script uses per-client cosine values as a historical similarity proxy.")
    print("=" * 80)


if __name__ == "__main__":
    main()

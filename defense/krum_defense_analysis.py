#!/usr/bin/env python3
"""
Krum-Inspired Defense Analysis for Federated Learning

This module analyzes Euclidean distance and cosine similarity metrics (per round, per client)
to detect Byzantine attacks (e.g., ALIE, Gaussian RMP) and compute detection rates.

NOTE: True Krum (Blanchard et al., NeurIPS 2017) requires pairwise distances between
client updates: ||Δ_i - Δ_j||. The provided metrics are:
  - Euclidean distance: ||Δ_i - Δ_g|| (each client vs weighted average)
  - Cosine similarity: cos(Δ_i, Δ_g)

Without pairwise distances, we implement:
  1. Krum-inspired heuristics using distance-to-mean as a proxy
  2. Statistical outlier detection (ALIE: attackers have LOW distance, HIGH cosine)
  3. Combined detection rules for attack presence and attacker identification
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Data Parsing
# =============================================================================

def parse_metric_table(text: str, metric_name: str = "metric") -> Tuple[Dict[int, Dict[int, float]], List[int]]:
    """
    Parse a metric table in the format:
        Round | Client0(B) | Client1(B) | ... | Client6(A) | Mean | Std
        --------------------------------------------------------------------------------
        1      | 1.310655       | 1.275939       | ...
    
    Returns:
        data: {round: {client_id: value}}
        client_ids: [0, 1, 2, ...] (in order of columns)
    """
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if len(lines) < 2:
        return {}, []
    
    # Parse header to get client IDs
    header = lines[0]
    # Match "Client0(B)" or "Client5(A)" etc.
    client_pattern = re.compile(r'Client(\d+)\s*\([BA]\)')
    client_matches = client_pattern.findall(header)
    client_ids = [int(m) for m in client_matches]
    num_clients = len(client_ids)
    
    data = {}
    for line in lines[1:]:
        if line.startswith('-'):
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < num_clients + 2:  # round + clients + mean + std
            continue
        try:
            round_num = int(parts[0].strip())
        except ValueError:
            continue
        data[round_num] = {}
        for i, cid in enumerate(client_ids):
            idx = i + 1  # 0 is round
            if idx < len(parts):
                try:
                    val = float(parts[idx].strip())
                    data[round_num][cid] = val
                except ValueError:
                    pass
    return data, client_ids


def parse_attacker_ids_from_header(header_line: str) -> List[int]:
    """Extract attacker client IDs from header (A) vs benign (B)."""
    client_pattern = re.compile(r'Client(\d+)\s*\(([BA])\)')
    attackers = []
    for m in client_pattern.finditer(header_line):
        cid, label = int(m.group(1)), m.group(2)
        if label == 'A':
            attackers.append(cid)
    return attackers


# =============================================================================
# Krum-Inspired Detection
# =============================================================================

def krum_score_proxy_from_distance(
    distances: Dict[int, float],
    client_ids: List[int],
    num_byzantine: int
) -> Dict[int, float]:
    """
    Krum proxy: In true Krum, we select the client with smallest sum of squared
    distances to its k nearest neighbors (k = n - f - 1).
    
    Without pairwise distances, we use distance-to-mean as a proxy:
    - For standard Byzantine (outliers): attackers have LARGE distance → exclude high-distance clients
    - For ALIE: attackers have SMALL distance (designed to be plausible) → exclude low-distance clients
    
    We compute a "Krum-like" score: for ALIE, higher distance = more likely benign.
    Clients with smallest distance are flagged as suspicious (ALIE pattern).
    """
    n = len(client_ids)
    f = min(num_byzantine, n - 1)
    k = n - f - 1  # Krum uses k nearest neighbors
    k = max(1, k)
    
    dist_list = [(cid, distances.get(cid, float('inf'))) for cid in client_ids]
    dist_list.sort(key=lambda x: x[1])
    
    # Proxy: "inverse Krum" for ALIE - clients with smallest distance get lowest "trust" score
    # Trust score = rank by distance (higher distance = more trusted in ALIE)
    scores = {}
    for rank, (cid, d) in enumerate(dist_list):
        scores[cid] = d  # Use raw distance: in ALIE, low d = attacker
    return scores


def detect_attackers_low_distance(
    distances: Dict[int, float],
    threshold_percentile: float = 25.0
) -> List[int]:
    """
    ALIE heuristic: Attackers have significantly LOWER Euclidean distance to mean.
    Flag clients with distance below the given percentile.
    """
    vals = list(distances.values())
    if len(vals) < 2:
        return []
    threshold = np.percentile(vals, threshold_percentile)
    return [cid for cid, d in distances.items() if d <= threshold]


def detect_attackers_high_cosine(
    similarities: Dict[int, float],
    threshold_percentile: float = 75.0
) -> List[int]:
    """
    ALIE heuristic: Attackers have HIGHER cosine similarity to mean.
    Flag clients with cosine above the given percentile.
    """
    vals = list(similarities.values())
    if len(vals) < 2:
        return []
    threshold = np.percentile(vals, threshold_percentile)
    return [cid for cid, s in similarities.items() if s >= threshold]


def detect_attackers_combined(
    distances: Dict[int, float],
    similarities: Dict[int, float],
    dist_percentile: float = 25.0,
    cos_percentile: float = 75.0,
    mode: str = "and"
) -> List[int]:
    """
    Combined: ALIE attackers have LOW distance AND HIGH cosine.
    mode: "and" = must satisfy both; "or" = either
    """
    low_dist = set(detect_attackers_low_distance(distances, dist_percentile))
    high_cos = set(detect_attackers_high_cosine(similarities, cos_percentile))
    if mode == "and":
        return list(low_dist & high_cos)
    return list(low_dist | high_cos)


def detect_attackers_krum_proxy(
    distances: Dict[int, float],
    client_ids: List[int],
    num_byzantine: int,
    num_to_flag: Optional[int] = None
) -> List[int]:
    """
    Krum-inspired: Flag the num_byzantine clients with SMALLEST distance (ALIE pattern).
    In true Krum we EXCLUDE these from aggregation; here we FLAG them as attackers.
    """
    if num_to_flag is None:
        num_to_flag = num_byzantine
    dist_list = [(cid, distances.get(cid, float('inf'))) for cid in client_ids]
    dist_list.sort(key=lambda x: x[1])
    return [cid for cid, _ in dist_list[:num_to_flag]]


def detect_attackers_mad(
    distances: Dict[int, float],
    k: float = 2.0
) -> List[int]:
    """
    Median Absolute Deviation: Flag clients with distance < median - k*MAD.
    ALIE: attackers have unusually low distance.
    """
    vals = np.array([distances[cid] for cid in distances])
    if len(vals) < 2:
        return []
    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    if mad < 1e-10:
        return []
    threshold = median - k * mad
    return [cid for cid, d in distances.items() if d <= threshold]


# =============================================================================
# Detection Rate Computation
# =============================================================================

@dataclass
class DetectionResult:
    """Result of defense analysis for one round."""
    round_num: int
    detected_attackers: List[int]
    true_attackers: List[int]
    benign_clients: List[int]
    false_positives: List[int]
    true_positives: List[int]
    false_negatives: List[int]


def analyze_round(
    round_num: int,
    distances: Dict[int, float],
    similarities: Dict[int, float],
    client_ids: List[int],
    attacker_ids: List[int],
    method: str = "combined",
    **kwargs
) -> DetectionResult:
    """Run detection for one round and compute TP/FP/FN."""
    benign_ids = [c for c in client_ids if c not in attacker_ids]
    
    if method == "low_distance":
        detected = detect_attackers_low_distance(distances, kwargs.get("percentile", 25.0))
    elif method == "high_cosine":
        detected = detect_attackers_high_cosine(similarities, kwargs.get("percentile", 75.0))
    elif method == "combined":
        detected = detect_attackers_combined(
            distances, similarities,
            kwargs.get("dist_percentile", 25.0),
            kwargs.get("cos_percentile", 75.0),
            kwargs.get("mode", "and")
        )
    elif method == "krum_proxy":
        detected = detect_attackers_krum_proxy(
            distances, client_ids,
            kwargs.get("num_byzantine", len(attacker_ids)),
            kwargs.get("num_to_flag", None)
        )
    elif method == "mad":
        detected = detect_attackers_mad(distances, kwargs.get("k", 2.0))
    else:
        detected = []
    
    true_positives = [c for c in detected if c in attacker_ids]
    false_positives = [c for c in detected if c in benign_ids]
    false_negatives = [c for c in attacker_ids if c not in detected]
    
    return DetectionResult(
        round_num=round_num,
        detected_attackers=detected,
        true_attackers=attacker_ids,
        benign_clients=benign_ids,
        false_positives=false_positives,
        true_positives=true_positives,
        false_negatives=false_negatives
    )


def compute_detection_rates(results: List[DetectionResult]) -> Dict[str, float]:
    """Compute aggregate detection metrics."""
    if not results:
        return {}
    
    total_rounds = len(results)
    total_attacker_rounds = sum(len(r.true_attackers) for r in results)
    total_tp = sum(len(r.true_positives) for r in results)
    total_fp = sum(len(r.false_positives) for r in results)
    total_fn = sum(len(r.false_negatives) for r in results)
    
    # Attack presence: rounds where we detected at least one attacker
    if total_attacker_rounds > 0:
        rounds_with_detection = sum(1 for r in results if len(r.true_positives) > 0)
        attack_presence_detection_rate = rounds_with_detection / total_rounds if total_rounds else 0.0
    else:
        # All-benign: rounds where we falsely detected (any detection is FP)
        rounds_with_detection = sum(1 for r in results if len(r.detected_attackers) > 0)
        attack_presence_detection_rate = rounds_with_detection / total_rounds if total_rounds else 0.0
    
    # Attacker identification: of all (attacker, round) pairs, how many did we flag?
    attacker_identification_rate = total_tp / total_attacker_rounds if total_attacker_rounds else 0.0
    
    # Precision and recall for attacker identification
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
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

def run_defense_analysis(
    euclidean_text: str,
    cosine_text: str,
    attacker_ids: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,
    **kwargs
) -> Dict:
    """
    Run full defense analysis on provided metric tables.
    
    Args:
        euclidean_text: Raw text of Euclidean distance table
        cosine_text: Raw text of cosine similarity table
        attacker_ids: List of attacker client IDs (if None, inferred from header)
        methods: Detection methods to run: ["low_distance", "high_cosine", "combined", "krum_proxy", "mad"]
    
    Returns:
        Dict with per-method results and detection rates
    """
    dist_data, client_ids = parse_metric_table(euclidean_text, "euclidean")
    cos_data, _ = parse_metric_table(cosine_text, "cosine")
    
    if not dist_data or not cos_data:
        return {"error": "Failed to parse data"}
    
    # Infer attacker IDs from cosine header if not provided
    if attacker_ids is None:
        first_line = cosine_text.strip().split('\n')[0]
        attacker_ids = parse_attacker_ids_from_header(first_line)
    
    if methods is None:
        methods = ["combined", "krum_proxy", "low_distance", "high_cosine", "mad"]
    
    rounds = sorted(set(dist_data.keys()) & set(cos_data.keys()))
    
    all_results = {}
    for method in methods:
        results = []
        for r in rounds:
            res = analyze_round(
                r, dist_data[r], cos_data[r],
                client_ids, attacker_ids,
                method=method,
                num_byzantine=len(attacker_ids),
                **kwargs
            )
            results.append(res)
        rates = compute_detection_rates(results)
        all_results[method] = {
            "per_round_results": results,
            "detection_rates": rates
        }
    
    return {
        "client_ids": client_ids,
        "attacker_ids": attacker_ids,
        "rounds": rounds,
        "methods": all_results
    }


def print_analysis_report(analysis: Dict) -> None:
    """Print a human-readable analysis report."""
    print("\n" + "=" * 80)
    print("KRUM-INSPIRED DEFENSE ANALYSIS REPORT")
    print("=" * 80)
    print(f"Clients: {analysis['client_ids']}")
    print(f"Attackers (ground truth): {analysis['attacker_ids']}")
    print(f"Rounds: {len(analysis['rounds'])}")
    if not analysis['attacker_ids']:
        print("  [All-benign scenario: Attack Presence Detection Rate = False Positive Rate]")
    print("=" * 80)
    
    for method, data in analysis["methods"].items():
        rates = data["detection_rates"]
        print(f"\n--- Method: {method.upper()} ---")
        if analysis['attacker_ids']:
            print(f"  Attack Presence Detection Rate: {rates['attack_presence_detection_rate']:.2%}")
            print(f"  Attacker Identification Rate:   {rates['attacker_identification_rate']:.2%}")
        else:
            print(f"  False Positive Rate (rounds with false detection): {rates['attack_presence_detection_rate']:.2%}")
        print(f"  Precision: {rates['precision']:.2%}  Recall: {rates['recall']:.2%}  F1: {rates['f1_score']:.2%}")
        print(f"  TP={rates['total_true_positives']} FP={rates['total_false_positives']} FN={rates['total_false_negatives']}")


# =============================================================================
# CLI with Embedded Data (Attack Scenario: Client5, Client6 = Attackers)
# =============================================================================

EMBEDDED_EUCLIDEAN_TABLE = """
Round | Client0(B) | Client1(B) | Client2(B) | Client3(B) | Client4(B) | Client5(A) | Client6(A) | Mean | Std
--------------------------------------------------------------------------------
1      | 1.325803       | 1.052455       | 1.071339       | 1.143043       | 1.098346       | 0.849862       | 1.161633       | 1.100354 | 0.132175
2      | 0.820074       | 0.832874       | 0.871093       | 0.756847       | 0.793252       | 0.512861       | 0.731688       | 0.759813 | 0.109711
3      | 0.639287       | 0.590164       | 0.645640       | 0.598202       | 0.618850       | 0.404241       | 0.556180       | 0.578938 | 0.076736
4      | 0.570667       | 0.513015       | 0.532351       | 0.570551       | 0.566703       | 0.373556       | 0.483532       | 0.515768 | 0.065635
5      | 0.583101       | 0.496775       | 0.521751       | 0.553853       | 0.566234       | 0.346313       | 0.489479       | 0.508215 | 0.073615
6      | 0.561013       | 0.495470       | 0.521042       | 0.537136       | 0.553502       | 0.344763       | 0.461886       | 0.496402 | 0.069578
7      | 0.551118       | 0.514541       | 0.550911       | 0.513902       | 0.523192       | 0.343989       | 0.462055       | 0.494244 | 0.067289
8      | 0.528959       | 0.497130       | 0.533465       | 0.502453       | 0.526645       | 0.284323       | 0.449360       | 0.474619 | 0.082162
9      | 0.523725       | 0.502335       | 0.525996       | 0.488252       | 0.518701       | 0.348460       | 0.471326       | 0.482685 | 0.057842
10     | 0.506849       | 0.510261       | 0.553933       | 0.470606       | 0.497151       | 0.305550       | 0.425211       | 0.467080 | 0.075357
11     | 0.497126       | 0.500907       | 0.522957       | 0.439873       | 0.473341       | 0.292409       | 0.429931       | 0.450935 | 0.071728
12     | 0.483505       | 0.494265       | 0.511436       | 0.438088       | 0.465401       | 0.295033       | 0.398918       | 0.440949 | 0.068958
13     | 0.452826       | 0.463857       | 0.499049       | 0.408811       | 0.438445       | 0.291076       | 0.410267       | 0.423476 | 0.061350
14     | 0.452753       | 0.466953       | 0.511801       | 0.409340       | 0.430128       | 0.283627       | 0.374651       | 0.418465 | 0.068200
15     | 0.440098       | 0.490484       | 0.520326       | 0.405112       | 0.411242       | 0.298090       | 0.431908       | 0.428180 | 0.065760
16     | 0.428554       | 0.470459       | 0.492389       | 0.410011       | 0.412936       | 0.271484       | 0.397426       | 0.411894 | 0.065592
17     | 0.419694       | 0.461940       | 0.479968       | 0.404902       | 0.407817       | 0.303798       | 0.421252       | 0.414196 | 0.052099
18     | 0.414141       | 0.475280       | 0.498042       | 0.383742       | 0.382748       | 0.304061       | 0.420716       | 0.411247 | 0.059509
19     | 0.394330       | 0.463623       | 0.480296       | 0.374074       | 0.369771       | 0.297476       | 0.428569       | 0.401163 | 0.057856
20     | 0.389982       | 0.450301       | 0.473386       | 0.373026       | 0.394652       | 0.298935       | 0.418831       | 0.399873 | 0.052564
21     | 0.398430       | 0.470201       | 0.508129       | 0.399773       | 0.368687       | 0.294047       | 0.438382       | 0.411093 | 0.064852
22     | 0.370369       | 0.420751       | 0.449906       | 0.344862       | 0.355007       | 0.277382       | 0.362000       | 0.368611 | 0.051334
23     | 0.347133       | 0.420030       | 0.469089       | 0.365251       | 0.362429       | 0.301590       | 0.404411       | 0.381419 | 0.050534
24     | 0.383628       | 0.471037       | 0.513054       | 0.399052       | 0.375259       | 0.297736       | 0.442395       | 0.411737 | 0.065465
25     | 0.372818       | 0.449096       | 0.481063       | 0.362071       | 0.368630       | 0.284155       | 0.396889       | 0.387817 | 0.059202
26     | 0.398020       | 0.450072       | 0.462666       | 0.371471       | 0.385451       | 0.255425       | 0.401561       | 0.389238 | 0.062744
27     | 0.389940       | 0.472309       | 0.497148       | 0.398960       | 0.399087       | 0.287489       | 0.404816       | 0.407107 | 0.062218
28     | 0.394228       | 0.448781       | 0.481017       | 0.348526       | 0.381965       | 0.277561       | 0.383238       | 0.387903 | 0.061178
29     | 0.390855       | 0.454619       | 0.482876       | 0.414976       | 0.351140       | 0.277390       | 0.412752       | 0.397801 | 0.062916
30     | 0.419940       | 0.469918       | 0.485662       | 0.394126       | 0.413775       | 0.287675       | 0.415145       | 0.412320 | 0.059320
31     | 0.412611       | 0.463884       | 0.499468       | 0.417370       | 0.389657       | 0.292362       | 0.391174       | 0.409504 | 0.060444
32     | 0.368281       | 0.445622       | 0.480970       | 0.360688       | 0.360554       | 0.287824       | 0.408140       | 0.387440 | 0.058907
33     | 0.378286       | 0.463611       | 0.482166       | 0.351718       | 0.367992       | 0.270694       | 0.411183       | 0.389379 | 0.066203
34     | 0.385565       | 0.482391       | 0.505011       | 0.381785       | 0.381065       | 0.270395       | 0.386649       | 0.398980 | 0.071397
35     | 0.373397       | 0.432979       | 0.464628       | 0.387331       | 0.360844       | 0.303917       | 0.400458       | 0.389079 | 0.047903
36     | 0.374940       | 0.477858       | 0.499118       | 0.365440       | 0.375371       | 0.264760       | 0.372147       | 0.389948 | 0.072382
37     | 0.368862       | 0.451794       | 0.475775       | 0.348809       | 0.367944       | 0.285686       | 0.393666       | 0.384648 | 0.059161
38     | 0.360036       | 0.464446       | 0.494725       | 0.391327       | 0.373024       | 0.279817       | 0.433314       | 0.399527 | 0.066632
39     | 0.356777       | 0.456083       | 0.466248       | 0.395602       | 0.379789       | 0.258507       | 0.337974       | 0.378711 | 0.066036
40     | 0.382086       | 0.468227       | 0.496886       | 0.404548       | 0.365680       | 0.301097       | 0.399852       | 0.402625 | 0.060113
41     | 0.408983       | 0.485320       | 0.501383       | 0.385930       | 0.390629       | 0.286515       | 0.388304       | 0.406723 | 0.066142
42     | 0.376737       | 0.450880       | 0.483815       | 0.393582       | 0.361576       | 0.254121       | 0.394165       | 0.387839 | 0.067562
43     | 0.404433       | 0.484819       | 0.522676       | 0.340954       | 0.371932       | 0.296818       | 0.404977       | 0.403801 | 0.072809
44     | 0.367350       | 0.445095       | 0.470315       | 0.392500       | 0.368723       | 0.278964       | 0.410614       | 0.390509 | 0.057582
45     | 0.374147       | 0.475849       | 0.512203       | 0.365998       | 0.362163       | 0.290375       | 0.435672       | 0.402344 | 0.070560
46     | 0.328699       | 0.421695       | 0.461013       | 0.322496       | 0.326856       | 0.278344       | 0.383445       | 0.360364 | 0.059415
47     | 0.392443       | 0.447208       | 0.479586       | 0.414854       | 0.382536       | 0.287538       | 0.386585       | 0.398679 | 0.056011
48     | 0.345893       | 0.442440       | 0.493732       | 0.320761       | 0.360889       | 0.274912       | 0.346747       | 0.369339 | 0.068895
49     | 0.368173       | 0.429584       | 0.466630       | 0.335813       | 0.340773       | 0.289709       | 0.397509       | 0.375456 | 0.055945
50     | 0.395150       | 0.450967       | 0.470441       | 0.403500       | 0.381398       | 0.289624       | 0.404076       | 0.399308 | 0.053636
"""

EMBEDDED_COSINE_TABLE = """
Round | Client0(B) | Client1(B) | Client2(B) | Client3(B) | Client4(B) | Client5(A) | Client6(A) | Mean | Std
--------------------------------------------------------------------------------
1      | 0.444424       | 0.432597       | 0.415358       | 0.530573       | 0.529018       | 0.454388       | 0.445578       | 0.464562 | 0.042802
2      | 0.564205       | 0.562710       | 0.553646       | 0.580225       | 0.576829       | 0.615833       | 0.563539       | 0.573855 | 0.019063
3      | 0.655470       | 0.686989       | 0.682861       | 0.647103       | 0.679302       | 0.668226       | 0.670468       | 0.670060 | 0.013519
4      | 0.389081       | 0.393489       | 0.374519       | 0.333572       | 0.393804       | 0.377904       | 0.378179       | 0.377221 | 0.019251
5      | 0.215400       | 0.196391       | 0.186027       | 0.188045       | 0.228652       | 0.276762       | 0.120165       | 0.201634 | 0.044170
6      | 0.124851       | 0.044557       | 0.015845       | 0.112626       | 0.145702       | 0.180045       | 0.100249       | 0.103411 | 0.052573
7      | 0.156528       | 0.088449       | 0.069461       | 0.156668       | 0.191371       | 0.119359       | 0.131171       | 0.130430 | 0.039072
8      | 0.142406       | 0.093958       | 0.083188       | 0.119860       | 0.179346       | 0.224318       | 0.077368       | 0.131492 | 0.050473
9      | 0.120591       | 0.051964       | 0.051632       | 0.099841       | 0.141370       | 0.037456       | 0.091975       | 0.084975 | 0.036211
10     | 0.113069       | 0.037493       | 0.031086       | 0.089826       | 0.139286       | 0.180990       | 0.090101       | 0.097407 | 0.049403
11     | 0.188629       | 0.171279       | 0.175575       | 0.160058       | 0.204101       | 0.222239       | 0.193346       | 0.187889 | 0.019549
12     | 0.080356       | 0.024152       | 0.010560       | 0.069600       | 0.103685       | 0.159659       | 0.072907       | 0.074417 | 0.045963
13     | 0.097469       | 0.051559       | 0.048853       | 0.075896       | 0.109297       | 0.094745       | 0.058301       | 0.076588 | 0.022571
14     | 0.065399       | 0.007999       | -0.001192      | 0.055197       | 0.095702       | 0.100416       | 0.047036       | 0.052937 | 0.036307
15     | 0.086932       | -0.009769      | -0.012830      | 0.076768       | 0.113936       | 0.134529       | 0.044503       | 0.062010 | 0.053257
16     | 0.067358       | 0.005127       | 0.011814       | 0.064128       | 0.085159       | 0.123281       | 0.024794       | 0.054523 | 0.039769
17     | 0.071763       | 0.027036       | 0.031745       | 0.063903       | 0.085772       | 0.016004       | 0.030220       | 0.046635 | 0.024712
18     | 0.075831       | -0.011729      | -0.013333      | 0.068490       | 0.080617       | 0.072500       | 0.017115       | 0.041356 | 0.039339
19     | 0.109591       | 0.063633       | 0.057811       | 0.094817       | 0.111731       | 0.099850       | 0.083598       | 0.088719 | 0.019774
20     | 0.103830       | 0.056708       | 0.066952       | 0.083593       | 0.111202       | 0.082046       | 0.058582       | 0.080416 | 0.019744
21     | 0.054470       | -0.023956      | -0.035838      | 0.060211       | 0.057520       | 0.108508       | 0.009991       | 0.032987 | 0.047833
22     | 0.284100       | 0.323648       | 0.327042       | 0.259267       | 0.283168       | 0.290045       | 0.291941       | 0.294173 | 0.022078
23     | 0.106177       | 0.107397       | 0.105777       | 0.085107       | 0.107387       | 0.081588       | 0.101528       | 0.099280 | 0.010284
24     | 0.041367       | -0.008782      | -0.025220      | 0.061757       | 0.065809       | 0.078425       | 0.002153       | 0.030787 | 0.037965
25     | 0.085579       | 0.044434       | 0.053125       | 0.067484       | 0.104636       | 0.124316       | 0.072063       | 0.078805 | 0.026183
26     | 0.067828       | 0.000957       | 0.004836       | 0.054221       | 0.093043       | 0.090483       | 0.011847       | 0.046174 | 0.037082
27     | 0.041736       | -0.030249      | -0.032521      | 0.052048       | 0.068877       | 0.110269       | 0.011802       | 0.031709 | 0.048444
28     | 0.083739       | 0.024610       | 0.030016       | 0.056808       | 0.125358       | 0.127876       | 0.041244       | 0.069950 | 0.040112
29     | 0.063726       | -0.000742      | 0.005521       | 0.094374       | 0.061418       | 0.047116       | 0.037814       | 0.044175 | 0.031077
30     | 0.049481       | -0.029804      | -0.025046      | 0.024018       | 0.073092       | 0.043673       | 0.027358       | 0.023253 | 0.035363
31     | 0.036254       | -0.025329      | -0.030519      | 0.041792       | 0.046944       | 0.083402       | 0.033787       | 0.026619 | 0.037749
32     | 0.085099       | 0.073158       | 0.082123       | 0.074313       | 0.100943       | 0.130972       | 0.072483       | 0.088442 | 0.019654
33     | 0.050484       | -0.014209      | -0.000653      | 0.036620       | 0.075813       | 0.092850       | 0.023791       | 0.037813 | 0.035892
34     | 0.050389       | -0.045072      | -0.047965      | 0.074879       | 0.101926       | 0.124831       | 0.017160       | 0.039450 | 0.063076
35     | 0.182140       | 0.187258       | 0.199507       | 0.164488       | 0.199188       | 0.181863       | 0.180560       | 0.185001 | 0.011182
36     | 0.034474       | -0.049817      | -0.054907      | 0.056033       | 0.087266       | 0.098525       | 0.022903       | 0.027782 | 0.056402
37     | 0.072100       | 0.036199       | 0.047023       | 0.050033       | 0.087647       | 0.118349       | 0.040762       | 0.064588 | 0.027703
38     | 0.038662       | -0.013319      | -0.005628      | 0.079814       | 0.093077       | 0.042632       | 0.021696       | 0.036705 | 0.037023
39     | 0.056666       | 0.027210       | 0.036249       | 0.066438       | 0.086515       | 0.138716       | 0.070461       | 0.068894 | 0.034115
40     | 0.033340       | -0.000810      | -0.001535      | 0.062920       | 0.054025       | 0.073466       | 0.034695       | 0.036586 | 0.027317
41     | 0.051282       | -0.028158      | -0.026633      | 0.055179       | 0.083154       | 0.100649       | 0.011178       | 0.035236 | 0.047292
42     | 0.087747       | 0.028656       | 0.045731       | 0.082423       | 0.099182       | 0.080579       | 0.041014       | 0.066476 | 0.025310
43     | 0.084996       | -0.036998      | -0.045920      | 0.052921       | 0.109720       | 0.113416       | 0.035379       | 0.044788 | 0.060491
44     | 0.137984       | 0.136131       | 0.155297       | 0.101680       | 0.150222       | 0.199531       | 0.126858       | 0.143958 | 0.027859
45     | 0.080819       | -0.003466      | -0.005323      | 0.078688       | 0.118500       | 0.128032       | 0.049157       | 0.063772 | 0.049492
46     | 0.205360       | 0.207430       | 0.224459       | 0.164799       | 0.201646       | 0.183672       | 0.077715       | 0.180726 | 0.045581
47     | 0.077526       | 0.034665       | 0.054563       | 0.058128       | 0.082553       | 0.128668       | 0.067433       | 0.071934 | 0.027424
48     | 0.043764       | -0.009393      | -0.019630      | 0.034813       | 0.105903       | 0.079151       | 0.037890       | 0.038928 | 0.041246
49     | 0.111862       | 0.089042       | 0.106118       | 0.067194       | 0.122450       | 0.138227       | 0.053081       | 0.098282 | 0.028083
50     | 0.060876       | 0.012644       | 0.023678       | 0.065582       | 0.071195       | 0.118818       | 0.042230       | 0.056432 | 0.032526
"""


def main():
    """Run defense analysis on embedded data (attack scenario: Client5, Client6 = attackers)."""
    print("Running Krum-inspired defense analysis on ATTACK scenario data...")
    print("(Client5, Client6 = attackers; Client0-4 = benign)")
    analysis = run_defense_analysis(
        EMBEDDED_EUCLIDEAN_TABLE,
        EMBEDDED_COSINE_TABLE,
        attacker_ids=None,  # Infer from header: all (B) -> no attackers
        methods=["combined", "krum_proxy", "low_distance", "high_cosine", "mad"]
    )
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return
    print_analysis_report(analysis)
    print("\n" + "=" * 80)
    print("Note: True Krum requires pairwise distances ||Δ_i - Δ_j|| between client updates.")
    print("This analysis uses distance-to-mean and cosine-to-mean as proxies.")
    print("=" * 80)


if __name__ == "__main__":
    main()

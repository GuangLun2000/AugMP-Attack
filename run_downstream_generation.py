#!/usr/bin/env python3
"""
Load a FedLLM SeqCLS checkpoint (NewsClassifierModel), transfer backbone to CausalLM, run generate() on probes.

Example:
  python run_downstream_generation.py \\
    --checkpoint results/global_checkpoint \\
    --probes data/ag_news_simple_probes.json \\
    --output results/downstream_gen.jsonl \\
    --stable --write-seq-cls-argmax

  python run_downstream_generation.py \\
    --checkpoint results/global_checkpoint \\
    --probes data/ag_news_curated_10.json \\
    --output results/downstream_curated.jsonl \\
    --prompt-style strict --stable --write-seq-cls-argmax --parse-strict-output

  python run_downstream_generation.py \\
    --checkpoint results/global_checkpoint \\
    --compare-checkpoint results_baseline/global_checkpoint \\
    --output results/downstream_compare.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decoder_adapters import resolve_adapter
from models import NewsClassifierModel

DEFAULT_MAX_NEW_TOKENS = 128
# Under --stable: enough for one AG News class label + one concise explanation (~20 words)
STABLE_MAX_NEW_TOKENS = 64
STABLE_REPETITION_PENALTY = 1.1
# Under --stable with --prompt-style strict: two-line Category + Reason template
STABLE_MAX_NEW_TOKENS_STRICT = 112
STABLE_REPETITION_PENALTY_STRICT = 1.15

AG_NEWS_ID2LABEL_FALLBACK = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

_STRICT_CATEGORY_CANONICAL = {
    "world": "World",
    "sports": "Sports",
    "business": "Business",
    "sci/tech": "Sci/Tech",
}
_STRICT_CATEGORY_LINE = re.compile(
    r"^\s*Category:\s*(World|Sports|Business|Sci/Tech)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_STRICT_REASON_LINE = re.compile(r"^\s*Reason:\s*(.+)$", re.MULTILINE)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_checkpoint_paths(checkpoint: Path) -> Tuple[Path, Path]:
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        return checkpoint / "global_model.pt", checkpoint / "checkpoint_metadata.json"
    return checkpoint, checkpoint.parent / "checkpoint_metadata.json"


def _load_metadata(pack: Dict[str, Any], meta_path: Path) -> Dict[str, Any]:
    meta = pack.get("metadata")
    if meta is None and meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    if not meta:
        raise ValueError("Missing metadata: expected 'metadata' in .pt or checkpoint_metadata.json next to checkpoint.")
    return meta


def build_news_classifier(meta: Dict[str, Any]) -> NewsClassifierModel:
    use_lora = bool(meta.get("use_lora", False))
    kw: Dict[str, Any] = {
        "model_name": meta["model_name"],
        "num_labels": int(meta["num_labels"]),
        "use_lora": use_lora,
    }
    if use_lora:
        kw["lora_r"] = meta.get("lora_r", 16)
        kw["lora_alpha"] = meta.get("lora_alpha", 32)
        kw["lora_dropout"] = meta.get("lora_dropout", 0.1)
        tm = meta.get("lora_target_modules")
        kw["lora_target_modules"] = None if tm is None else list(tm)
    return NewsClassifierModel(**kw)


def load_news_classifier(pt_path: Path, meta_path: Path) -> Tuple[NewsClassifierModel, Dict[str, Any]]:
    pack = _torch_load(pt_path)
    meta = _load_metadata(pack, meta_path)
    model = build_news_classifier(meta)
    incompatible = model.load_state_dict(pack["state_dict"], strict=False)
    if incompatible.missing_keys:
        print(f"  Warning: missing_keys when loading NewsClassifierModel: {len(incompatible.missing_keys)} keys")
    if incompatible.unexpected_keys:
        print(f"  Warning: unexpected_keys when loading NewsClassifierModel: {len(incompatible.unexpected_keys)} keys")
    model.eval()
    return model, meta


def build_causal_lm(base_model_name: str, device: torch.device) -> torch.nn.Module:
    causal = AutoModelForCausalLM.from_pretrained(base_model_name)
    causal.to(device)
    causal.eval()
    return causal


def build_tokenizer(base_model_name: str):
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def default_prompt(news_text: str, question: str) -> str:
    return f"### News\n{news_text}\n\n### Task\n{question}\n\n### Answer\n"


def simple_prompt(news_text: str, question: str) -> str:
    """Shorter template aligned with easy AG News-style probes."""
    return f"News: {news_text}\n{question}\nAnswer:"


def strict_prompt(news_text: str, question: str) -> str:
    """Two-line Category / Reason template; `question` ignored (single global instruction)."""
    _ = question
    return (
        "You will classify one news article into exactly one AG News category.\n\n"
        "Categories (pick exactly one word): World, Sports, Business, Sci/Tech.\n\n"
        f"Article:\n{news_text}\n\n"
        "Rules:\n"
        "- Output EXACTLY two lines in English.\n"
        "- Line 1 must be: Category: <World|Sports|Business|Sci/Tech>\n"
        "- Line 2 must be: Reason: <one sentence, at most 25 words; explain using only evidence from the article>\n"
        "- Do not repeat the article. Do not ask questions. No extra lines before Line 1.\n\n"
        "Answer:\n"
    )


def parse_strict_completion(completion: str) -> Tuple[Optional[str], Optional[str], bool]:
    """Extract Category / Reason lines; return (category, reason, format_valid)."""
    text = (completion or "").strip()
    cm = _STRICT_CATEGORY_LINE.search(text)
    if not cm:
        return None, None, False
    key = cm.group(1).strip().lower()
    category = _STRICT_CATEGORY_CANONICAL.get(key)
    if category is None:
        return None, None, False
    tail = text[cm.end() :]
    rm = _STRICT_REASON_LINE.search(tail)
    if not rm:
        return category, None, False
    reason = rm.group(1).strip()
    if not reason:
        return category, None, False
    return category, reason, True


def make_prompt_fn(style: str) -> Callable[[str, str], str]:
    if style == "simple":
        return simple_prompt
    if style == "strict":
        return strict_prompt
    if style == "default":
        return default_prompt
    raise ValueError(f"Unknown prompt style: {style!r} (use 'default', 'simple', or 'strict')")


def load_probes(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Probes JSON must be a list of objects with id, news_text, and optional question")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "news_text" not in item:
            raise ValueError(f"Probe at index {i} must be an object with 'news_text'")
        if "question" not in item:
            item["question"] = ""
    return data


def _classifier_config_from_news(news: NewsClassifierModel):
    """Resolve HuggingFace PretrainedConfig for id2label (handles PEFT wrapper)."""
    m = news.model
    cfg = getattr(m, "config", None)
    if cfg is not None and getattr(cfg, "id2label", None):
        return cfg
    base = getattr(m, "base_model", None)
    if base is not None:
        inner = getattr(base, "model", base)
        cfg = getattr(inner, "config", None)
        if cfg is not None:
            return cfg
    return cfg


def get_id2label_map(news: NewsClassifierModel, num_labels: int) -> Dict[int, str]:
    cfg = _classifier_config_from_news(news)
    if cfg is not None and getattr(cfg, "id2label", None):
        return {int(k): str(v) for k, v in cfg.id2label.items()}
    if num_labels == 4:
        return dict(AG_NEWS_ID2LABEL_FALLBACK)
    return {i: str(i) for i in range(num_labels)}


def seq_cls_argmax_one(
    news: NewsClassifierModel,
    tokenizer,
    text: str,
    id2label: Dict[int, str],
    max_length: int = 512,
) -> Tuple[int, str]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    dev = next(news.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        logits = news(enc["input_ids"], enc["attention_mask"])
    pid = int(logits.argmax(dim=-1).item())
    label = id2label.get(pid, str(pid))
    return pid, label


def collect_seq_cls_predictions(
    news: NewsClassifierModel,
    tokenizer,
    probes: List[Dict[str, Any]],
    num_labels: int,
    max_length: int = 512,
) -> List[Tuple[int, str]]:
    id2label = get_id2label_map(news, num_labels)
    out: List[Tuple[int, str]] = []
    for p in probes:
        pid, lab = seq_cls_argmax_one(news, tokenizer, p["news_text"], id2label, max_length=max_length)
        out.append((pid, lab))
    return out


def generate_completion(
    causal,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: Optional[float] = None,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    gen_kw: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = max(temperature, 1e-5)
    else:
        gen_kw["do_sample"] = False
    if repetition_penalty is not None and repetition_penalty > 0:
        gen_kw["repetition_penalty"] = repetition_penalty

    with torch.no_grad():
        out = causal.generate(**inputs, **gen_kw)
    inp_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, inp_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def run_for_checkpoint(
    news: NewsClassifierModel,
    meta: Dict[str, Any],
    base_model_name: str,
    probes: List[Dict[str, Any]],
    device: torch.device,
    prompt_fn: Callable[[str, str], str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: Optional[float] = None,
) -> List[str]:
    adapter = resolve_adapter(meta["model_name"])
    causal = build_causal_lm(base_model_name, device)
    tokenizer = build_tokenizer(base_model_name)
    adapter.transfer_backbone(news.model, causal)

    completions = []
    for p in probes:
        prompt = prompt_fn(p["news_text"], p.get("question", ""))
        text = generate_completion(
            causal,
            tokenizer,
            prompt,
            device,
            max_new_tokens,
            do_sample,
            temperature,
            repetition_penalty=repetition_penalty,
        )
        completions.append(text)
    del causal
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return completions


def main() -> None:
    parser = argparse.ArgumentParser(description="Downstream causal LM generation from Fed SeqCLS checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to global_model.pt or directory containing global_model.pt + checkpoint_metadata.json",
    )
    parser.add_argument(
        "--compare-checkpoint",
        type=Path,
        default=None,
        help="Optional second checkpoint (e.g. clean baseline) for paired completions on the same probes",
    )
    parser.add_argument(
        "--probes",
        type=Path,
        default=Path("data/financial_probes.json"),
        help="JSON list of {id, news_text, question}",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override HF model id for CausalLM/tokenizer (default: model_name from checkpoint metadata)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=f"Max new tokens (default: {DEFAULT_MAX_NEW_TOKENS}, or {STABLE_MAX_NEW_TOKENS} when --stable)",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling (greedy decode)")
    parser.add_argument(
        "--stable",
        action="store_true",
        help="Greedy decode + short generations + repetition_penalty (recommended for short label-style answers)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Passed to generate(); when --stable and omitted, uses %.2f" % STABLE_REPETITION_PENALTY,
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=("default", "simple", "strict"),
        default="default",
        help="Prompt template: 'simple' short News/Answer; 'strict' two-line Category/Reason AG News format",
    )
    parser.add_argument(
        "--parse-strict-output",
        action="store_true",
        help="Parse completion_primary for strict Category/Reason lines; adds parsed_* and format_valid to JSONL",
    )
    parser.add_argument(
        "--write-seq-cls-argmax",
        action="store_true",
        help="Add seq_cls_label_id / seq_cls_label from the SeqCLS head (stable vs free generation)",
    )
    parser.add_argument(
        "--cls-max-length",
        type=int,
        default=512,
        help="Tokenizer max_length for --write-seq-cls-argmax",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    pt_path, meta_path = resolve_checkpoint_paths(args.checkpoint)
    if not pt_path.is_file():
        print(f"Checkpoint file not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    probes = load_probes(args.probes)
    news_a, meta_a = load_news_classifier(pt_path, meta_path)
    base_name = args.base_model or meta_a["model_name"]
    if args.base_model and args.base_model != meta_a["model_name"]:
        print(
            f"  Note: --base-model {args.base_model!r} overrides metadata model_name {meta_a['model_name']!r}; "
            "ensure architecture matches the saved weights."
        )

    do_sample = not args.greedy
    repetition_penalty: Optional[float] = args.repetition_penalty

    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    elif args.stable:
        max_new_tokens = (
            STABLE_MAX_NEW_TOKENS_STRICT if args.prompt_style == "strict" else STABLE_MAX_NEW_TOKENS
        )
    else:
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    if args.stable:
        do_sample = False
        if repetition_penalty is None:
            repetition_penalty = (
                STABLE_REPETITION_PENALTY_STRICT
                if args.prompt_style == "strict"
                else STABLE_REPETITION_PENALTY
            )

    prompt_fn = make_prompt_fn(args.prompt_style)
    tokenizer_shared = build_tokenizer(base_name)

    seq_primary: Optional[List[Tuple[int, str]]] = None
    seq_compare: Optional[List[Tuple[int, str]]] = None
    if args.write_seq_cls_argmax:
        nlab = int(meta_a["num_labels"])
        seq_primary = collect_seq_cls_predictions(
            news_a, tokenizer_shared, probes, nlab, max_length=args.cls_max_length
        )
        print(f"  SeqCLS argmax predictions (primary): {len(seq_primary)} probes")

    print(f"  Primary checkpoint: {pt_path}")
    completions_a = run_for_checkpoint(
        news_a,
        meta_a,
        base_name,
        probes,
        device,
        prompt_fn,
        max_new_tokens,
        do_sample,
        args.temperature,
        repetition_penalty=repetition_penalty,
    )

    completions_b: Optional[List[str]] = None
    news_b = None
    meta_b = None
    if args.compare_checkpoint:
        pt_b, meta_b_path = resolve_checkpoint_paths(args.compare_checkpoint)
        if not pt_b.is_file():
            print(f"Compare checkpoint not found: {pt_b}", file=sys.stderr)
            sys.exit(1)
        news_b, meta_b = load_news_classifier(pt_b, meta_b_path)
        if meta_b["model_name"] != meta_a["model_name"]:
            print(
                "  Warning: compare checkpoint has different model_name than primary; "
                "verify adapters and base-model compatibility.",
                file=sys.stderr,
            )
        if args.write_seq_cls_argmax:
            nlab_b = int(meta_b["num_labels"])
            seq_compare = collect_seq_cls_predictions(
                news_b, tokenizer_shared, probes, nlab_b, max_length=args.cls_max_length
            )
            print(f"  SeqCLS argmax predictions (compare): {len(seq_compare)} probes")
        print(f"  Compare checkpoint: {pt_b}")
        completions_b = run_for_checkpoint(
            news_b,
            meta_b,
            base_name,
            probes,
            device,
            prompt_fn,
            max_new_tokens,
            do_sample,
            args.temperature,
            repetition_penalty=repetition_penalty,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, p in enumerate(probes):
            row: Dict[str, Any] = {
                "probe_id": p.get("id", i + 1),
                "news_text": p["news_text"],
                "question": p.get("question", ""),
                "prompt_style": args.prompt_style,
                "completion_primary": completions_a[i],
            }
            if "gold_ag_label" in p:
                row["gold_ag_label"] = p["gold_ag_label"]
            if "gold_category" in p:
                row["gold_category"] = p["gold_category"]
            if args.parse_strict_output:
                pc, pr, fv = parse_strict_completion(completions_a[i])
                row["parsed_category"] = pc
                row["parsed_reason"] = pr
                row["format_valid"] = fv
                if p.get("gold_category") is not None:
                    row["matches_gold_category"] = bool(fv and pc is not None and pc == p["gold_category"])
            if completions_b is not None:
                row["completion_compare"] = completions_b[i]
            if seq_primary is not None:
                row["seq_cls_label_id"] = seq_primary[i][0]
                row["seq_cls_label"] = seq_primary[i][1]
            if seq_compare is not None:
                row["seq_cls_compare_label_id"] = seq_compare[i][0]
                row["seq_cls_compare_label"] = seq_compare[i][1]
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(probes)} lines to {args.output}")


if __name__ == "__main__":
    main()

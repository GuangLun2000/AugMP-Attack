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
    --probes data/ag_news_curated_10.json \\
    --output results/downstream_twostage.jsonl \\
    --prompt-style strict --stable --write-seq-cls-argmax --parse-strict-output \\
    --strict-two-stage

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
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from decoder_adapters import resolve_adapter
from models import NewsClassifierModel

DEFAULT_MAX_NEW_TOKENS = 128
# Under --stable: enough for one AG News class label + one concise explanation (~20 words)
STABLE_MAX_NEW_TOKENS = 64
STABLE_REPETITION_PENALTY = 1.1
# Under --stable with --prompt-style strict: two-line Category + Reason + stopping criteria
STABLE_MAX_NEW_TOKENS_STRICT = 72
STABLE_REPETITION_PENALTY_STRICT = 1.15
# strict_json: compact JSON object
STABLE_MAX_NEW_TOKENS_STRICT_JSON = 96

AG_NEWS_ID2LABEL_FALLBACK = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
# Prompt ends with this; completion is stored as prefix + model continuation (full two-line answer).
STRICT_COMPLETION_PREFIX = "Category: "
STRICT_REASON_MAX_WORDS = 25
STRICT_PHASE_A_MAX_NEW_TOKENS = 16

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
_STRICT_LINE1_ANCHOR = re.compile(
    r"^Category:\s*(World|Sports|Business|Sci/Tech)\s*$",
    re.IGNORECASE,
)
_STRICT_LINE2_ANCHOR = re.compile(r"^Reason:\s*(.+)$", re.IGNORECASE)
_JSON_CATEGORY_KEYS = ("category", "label", "class")


class _StopAfterTwoLines(StoppingCriteria):
    """Stop once generated text has at least two lines (Category + Reason)."""

    def __init__(self, tokenizer: Any, prompt_token_len: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_token_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        gen_only = seq[self.prompt_len :]
        if gen_only.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True)
        return len(text.splitlines()) >= 2


class _StopAfterJsonBrace(StoppingCriteria):
    """Stop after first closing brace in generated text (best-effort for one JSON object)."""

    def __init__(self, tokenizer: Any, prompt_token_len: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_token_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        gen_only = seq[self.prompt_len :]
        if gen_only.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True)
        return "}" in text


class _StopAfterFirstNewline(StoppingCriteria):
    """Stop once generated text contains a newline (single-line continuation after 'Reason: ')."""

    def __init__(self, tokenizer: Any, prompt_token_len: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_token_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        gen_only = seq[self.prompt_len :]
        if gen_only.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True)
        return "\n" in text


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


_STRICT_FEW_SHOT_BLOCK = (
    "Example article:\n"
    "The central bank held rates steady while major indices closed mixed.\n\n"
    "Example answer (two lines only):\n"
    "Category: Business\n"
    "Reason: The piece concerns interest rates and stock markets.\n\n"
)


def strict_prompt(news_text: str, question: str, *, few_shot: bool = False) -> str:
    """Prefix completion: prompt ends with 'Category: '; model continues with '<Class>\\nReason: ...'."""
    _ = question
    fs = _STRICT_FEW_SHOT_BLOCK if few_shot else ""
    return (
        "You classify one news article into exactly one AG News category.\n\n"
        "Allowed category words (pick exactly one, never list multiple or use |): "
        "World, Sports, Business, Sci/Tech.\n\n"
        f"{fs}"
        f"Article:\n{news_text}\n\n"
        "Rules:\n"
        "- Your continuation completes line 1 after 'Category: ' with exactly ONE of the four words above.\n"
        "- Line 2 must start with 'Reason: ' then ONE English sentence, at most 25 words, citing only the article.\n"
        "- Do not repeat the article text. No extra lines. No questions.\n\n"
        "Begin your answer now (output nothing before the category word on line 1):\n"
        f"{STRICT_COMPLETION_PREFIX}"
    )


def strict_json_prompt(news_text: str, question: str) -> str:
    _ = question
    return (
        "Classify the article into exactly one AG News category.\n"
        'Output ONE JSON object only, no markdown fences, no extra text. Keys: "category" and "reason".\n'
        '"category" must be exactly one of: World, Sports, Business, Sci/Tech.\n'
        '"reason" must be one English sentence (max 25 words) using only evidence from the article.\n\n'
        f"Article:\n{news_text}\n\n"
        "JSON:\n"
    )


def _reason_word_count(reason: str) -> int:
    return len([w for w in (reason or "").split() if w])


def parse_strict_completion(completion: str) -> Tuple[Optional[str], Optional[str], bool, bool]:
    """Return (category, reason, format_valid, format_valid_strict).

    format_valid: first Category/Reason pair found anywhere (lenient).
    format_valid_strict: exactly two non-empty lines, anchored Category/Reason, reason <=25 words.
    """
    text = (completion or "").strip()
    if not text:
        return None, None, False, False

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 2:
        m1 = _STRICT_LINE1_ANCHOR.match(lines[0])
        m2 = _STRICT_LINE2_ANCHOR.match(lines[1])
        if m1 and m2:
            c2 = _STRICT_CATEGORY_CANONICAL.get(m1.group(1).strip().lower())
            r2 = m2.group(1).strip()
            if c2 and r2 and _reason_word_count(r2) <= STRICT_REASON_MAX_WORDS:
                return c2, r2, True, True

    cm = _STRICT_CATEGORY_LINE.search(text)
    if not cm:
        return None, None, False, False
    key = cm.group(1).strip().lower()
    category = _STRICT_CATEGORY_CANONICAL.get(key)
    if category is None:
        return None, None, False, False
    tail = text[cm.end() :]
    rm = _STRICT_REASON_LINE.search(tail)
    if not rm:
        return category, None, False, False
    reason = rm.group(1).strip()
    if not reason:
        return category, None, False, False
    return category, reason, True, False


def parse_strict_json_completion(completion: str) -> Tuple[Optional[str], Optional[str], bool, bool]:
    """Parse JSON style; format_valid_strict same as format_valid when JSON is valid and constraints hold."""
    raw = (completion or "").strip()
    if not raw:
        return None, None, False, False
    blob = raw
    if not blob.startswith("{"):
        i = raw.find("{")
        j = raw.rfind("}")
        if i >= 0 and j > i:
            blob = raw[i : j + 1]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None, None, False, False
    if not isinstance(obj, dict):
        return None, None, False, False
    cat_raw = None
    for k in _JSON_CATEGORY_KEYS:
        if k in obj:
            cat_raw = obj[k]
            break
    if cat_raw is None:
        return None, None, False, False
    reason = obj.get("reason")
    if not isinstance(cat_raw, str) or not isinstance(reason, str):
        return None, None, False, False
    key = cat_raw.strip().lower()
    category = _STRICT_CATEGORY_CANONICAL.get(key)
    if category is None:
        return None, None, False, False
    reason = reason.strip()
    if not reason:
        return category, None, False, False
    fv = True
    fvs = _reason_word_count(reason) <= STRICT_REASON_MAX_WORDS
    return category, reason, fv, fvs


def phase_a_category_prompt(news_text: str) -> str:
    return (
        "Pick exactly one AG News category for the article. "
        "Reply with ONE word only: World, Sports, Business, or Sci/Tech.\n\n"
        f"Article:\n{news_text}\n\n"
        "Answer:\n"
    )


def phase_b_reason_prompt(news_text: str, category: str) -> str:
    return (
        f"You already chose category: {category}.\n"
        "Write ONE English sentence (max 25 words) explaining why this article fits that category, "
        "using only evidence from the article. Do not repeat the whole article.\n\n"
        f"Article:\n{news_text}\n\n"
        "Output exactly one line starting with 'Reason: '.\n"
        "Reason: "
    )


def parse_phase_a_category(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    tl = t.lower()
    if re.search(r"\bsci\s*/\s*tech\b", tl) or "sci/tech" in tl.replace(" ", ""):
        return "Sci/Tech"
    for word, canon in (
        ("business", "Business"),
        ("sports", "Sports"),
        ("world", "World"),
    ):
        if re.search(rf"\b{word}\b", tl):
            return canon
    return None


def seq_cls_to_ag_category(label_id: int, label_str: str, num_labels: int) -> str:
    """Map SeqCLS prediction to canonical AG News category name."""
    if num_labels == 4:
        c = AG_NEWS_ID2LABEL_FALLBACK.get(label_id)
        if c:
            return c
    ls = (label_str or "").strip()
    lk = ls.lower()
    if lk in _STRICT_CATEGORY_CANONICAL:
        return _STRICT_CATEGORY_CANONICAL[lk]
    return ls if ls in ("World", "Sports", "Business", "Sci/Tech") else ""


def make_prompt_fn(style: str, *, strict_few_shot: bool = False) -> Callable[[str, str], str]:
    if style == "simple":
        return simple_prompt
    if style == "strict":
        return lambda nt, q: strict_prompt(nt, q, few_shot=strict_few_shot)
    if style == "strict_json":
        return strict_json_prompt
    if style == "default":
        return default_prompt
    raise ValueError(
        f"Unknown prompt style: {style!r} (use 'default', 'simple', 'strict', or 'strict_json')"
    )


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


def completion_parse_ok(
    completion: str,
    prompt_style: str,
    strict_two_stage: bool,
    lenient: bool,
) -> bool:
    """Whether completion satisfies retry success criterion (strict or lenient parse)."""
    if prompt_style == "strict_json" and not strict_two_stage:
        _, _, fv, fvs = parse_strict_json_completion(completion)
        return bool(fv if lenient else fvs)
    _, _, fv, fvs = parse_strict_completion(completion)
    return bool(fv if lenient else fvs)


def generate_completion(
    causal,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: Optional[float] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    inp_len = int(inputs["input_ids"].shape[1])
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
    if stopping_criteria is not None:
        gen_kw["stopping_criteria"] = stopping_criteria

    with torch.no_grad():
        out = causal.generate(**inputs, **gen_kw)
    gen_ids = out[0, inp_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _generate_strict_two_stage_one(
    causal,
    tokenizer,
    device: torch.device,
    news_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: Optional[float],
    *,
    reason_only: bool,
    seq_row: Optional[Tuple[int, str]],
    seq_primary: Optional[List[Tuple[int, str]]],
    probe_index: int,
    num_labels: int,
) -> str:
    if reason_only:
        if seq_row is None:
            raise ValueError("strict_two_stage with reason_only requires seq_cls row per probe.")
        pid, lab = seq_row
        cat = seq_cls_to_ag_category(pid, lab, num_labels)
    else:
        p1 = phase_a_category_prompt(news_text)
        inp1 = tokenizer(p1, return_tensors="pt", truncation=True, max_length=2048)
        plen1 = int(inp1["input_ids"].shape[1])
        stop1 = StoppingCriteriaList([_StopAfterFirstNewline(tokenizer, plen1)])
        raw_a = generate_completion(
            causal,
            tokenizer,
            p1,
            device,
            STRICT_PHASE_A_MAX_NEW_TOKENS,
            do_sample,
            temperature,
            repetition_penalty=repetition_penalty,
            stopping_criteria=stop1,
        )
        cat = parse_phase_a_category(raw_a)
        if not cat and seq_primary is not None:
            pid, lab = seq_primary[probe_index]
            cat = seq_cls_to_ag_category(pid, lab, num_labels)
        if not cat:
            cat = "World"

    p2 = phase_b_reason_prompt(news_text, cat)
    inp2 = tokenizer(p2, return_tensors="pt", truncation=True, max_length=2048)
    plen2 = int(inp2["input_ids"].shape[1])
    stop2 = StoppingCriteriaList([_StopAfterFirstNewline(tokenizer, plen2)])
    raw_b = generate_completion(
        causal,
        tokenizer,
        p2,
        device,
        max_new_tokens,
        do_sample,
        temperature,
        repetition_penalty=repetition_penalty,
        stopping_criteria=stop2,
    )
    rb = raw_b.strip()
    if rb.lower().startswith("reason:"):
        rb = rb.split(":", 1)[-1].strip()
    return f"Category: {cat}\nReason: {rb}"


def _generate_single_pass_one(
    causal,
    tokenizer,
    prompt_fn: Callable[[str, str], str],
    news_text: str,
    question: str,
    device: torch.device,
    prompt_style: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: Optional[float],
) -> str:
    prompt = prompt_fn(news_text, question)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    plen = int(inp["input_ids"].shape[1])
    stopping: Optional[StoppingCriteriaList] = None
    if prompt_style == "strict" and prompt.endswith(STRICT_COMPLETION_PREFIX):
        stopping = StoppingCriteriaList([_StopAfterTwoLines(tokenizer, plen)])
    elif prompt_style == "strict_json":
        stopping = StoppingCriteriaList([_StopAfterJsonBrace(tokenizer, plen)])

    raw = generate_completion(
        causal,
        tokenizer,
        prompt,
        device,
        max_new_tokens,
        do_sample,
        temperature,
        repetition_penalty=repetition_penalty,
        stopping_criteria=stopping,
    )
    if prompt_style == "strict" and prompt.endswith(STRICT_COMPLETION_PREFIX):
        return (STRICT_COMPLETION_PREFIX + raw.strip()).strip()
    return raw.strip()


def run_for_checkpoint(
    news: NewsClassifierModel,
    meta: Dict[str, Any],
    base_model_name: str,
    probes: List[Dict[str, Any]],
    device: torch.device,
    prompt_style: str,
    prompt_fn: Callable[[str, str], str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: Optional[float] = None,
    *,
    strict_two_stage: bool = False,
    reason_only: bool = False,
    seq_primary: Optional[List[Tuple[int, str]]] = None,
    num_labels: int = 4,
    parse_extra_retries: int = 0,
    retry_parse_lenient: bool = False,
    retry_temperature: float = 0.7,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    adapter = resolve_adapter(meta["model_name"])
    causal = build_causal_lm(base_model_name, device)
    tokenizer = build_tokenizer(base_model_name)
    adapter.transfer_backbone(news.model, causal)

    use_parse_retry = parse_extra_retries > 0 and (
        strict_two_stage or prompt_style in ("strict", "strict_json")
    )
    max_attempts = 1 + parse_extra_retries if use_parse_retry else 1

    completions: List[str] = []
    retry_meta: List[Dict[str, Any]] = []

    for i, p in enumerate(probes):
        news_text = p["news_text"]
        question = p.get("question", "")

        meta_row: Dict[str, Any] = {}
        if use_parse_retry:
            meta_row["parse_retry_max_attempts"] = max_attempts

        best_text = ""
        ok = False
        for attempt in range(max_attempts):
            att = attempt + 1
            sample_here = do_sample if attempt == 0 else True
            temp_here = temperature if attempt == 0 else retry_temperature

            if strict_two_stage:
                seq_row = (
                    seq_primary[i]
                    if reason_only and seq_primary is not None and i < len(seq_primary)
                    else None
                )
                text = _generate_strict_two_stage_one(
                    causal,
                    tokenizer,
                    device,
                    news_text,
                    max_new_tokens,
                    sample_here,
                    temp_here,
                    repetition_penalty,
                    reason_only=reason_only,
                    seq_row=seq_row,
                    seq_primary=seq_primary,
                    probe_index=i,
                    num_labels=num_labels,
                )
            else:
                text = _generate_single_pass_one(
                    causal,
                    tokenizer,
                    prompt_fn,
                    news_text,
                    question,
                    device,
                    prompt_style,
                    max_new_tokens,
                    sample_here,
                    temp_here,
                    repetition_penalty,
                )

            best_text = text
            if use_parse_retry:
                meta_row["parse_attempts"] = att
            if completion_parse_ok(
                text, prompt_style, strict_two_stage, retry_parse_lenient
            ):
                ok = True
                break

        if use_parse_retry:
            meta_row["parse_retry_exhausted"] = not ok

        completions.append(best_text)
        retry_meta.append(meta_row)

    del causal
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return completions, retry_meta


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
        help=(
            f"Max new tokens (default: {DEFAULT_MAX_NEW_TOKENS}; under --stable: 64 simple/default, "
            f"{STABLE_MAX_NEW_TOKENS_STRICT} strict, {STABLE_MAX_NEW_TOKENS_STRICT_JSON} strict_json)"
        ),
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
        choices=("default", "simple", "strict", "strict_json"),
        default="default",
        help="Prompt: 'strict' = prefix Category + two-line; 'strict_json' = one JSON object with category/reason",
    )
    parser.add_argument(
        "--strict-few-shot",
        action="store_true",
        help="With --prompt-style strict: prepend one in-prompt example (two-line format).",
    )
    parser.add_argument(
        "--strict-two-stage",
        action="store_true",
        help="Category via short phase-A prompt (or --reason-only from SeqCLS), then Reason only; implies two-line output.",
    )
    parser.add_argument(
        "--reason-only",
        action="store_true",
        help="With --strict-two-stage: fix category from SeqCLS head (--write-seq-cls-argmax required).",
    )
    parser.add_argument(
        "--parse-strict-output",
        action="store_true",
        help="Parse completion: strict lines or JSON; adds parsed_*, format_valid, format_valid_strict, matches_gold_*",
    )
    parser.add_argument(
        "--parse-retry-max",
        type=int,
        default=0,
        help=(
            "Extra decode attempts when parse fails (strict/strict_json/two-stage only). "
            "0 = single generation; 2 = up to 3 total attempts. Retries use sampling + --parse-retry-temperature."
        ),
    )
    parser.add_argument(
        "--parse-retry-lenient",
        action="store_true",
        help="Stop retrying on format_valid instead of format_valid_strict (looser).",
    )
    parser.add_argument(
        "--parse-retry-temperature",
        type=float,
        default=0.7,
        help="Temperature for retry attempts (2nd+); first attempt still follows --stable/--greedy.",
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
        if args.prompt_style == "strict_json":
            max_new_tokens = STABLE_MAX_NEW_TOKENS_STRICT_JSON
        elif args.prompt_style == "strict":
            max_new_tokens = STABLE_MAX_NEW_TOKENS_STRICT
        else:
            max_new_tokens = STABLE_MAX_NEW_TOKENS
    else:
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    if args.stable:
        do_sample = False
        if repetition_penalty is None:
            repetition_penalty = (
                STABLE_REPETITION_PENALTY_STRICT
                if args.prompt_style in ("strict", "strict_json")
                else STABLE_REPETITION_PENALTY
            )

    prompt_fn = make_prompt_fn(args.prompt_style, strict_few_shot=args.strict_few_shot)

    tokenizer_shared = build_tokenizer(base_name)

    seq_primary: Optional[List[Tuple[int, str]]] = None
    seq_compare: Optional[List[Tuple[int, str]]] = None
    if args.write_seq_cls_argmax:
        nlab = int(meta_a["num_labels"])
        seq_primary = collect_seq_cls_predictions(
            news_a, tokenizer_shared, probes, nlab, max_length=args.cls_max_length
        )
        print(f"  SeqCLS argmax predictions (primary): {len(seq_primary)} probes")

    if args.reason_only and not args.strict_two_stage:
        print("  Note: --reason-only has no effect without --strict-two-stage", file=sys.stderr)
    if args.strict_two_stage and args.reason_only and not args.write_seq_cls_argmax:
        print("  Error: --reason-only requires --write-seq-cls-argmax.", file=sys.stderr)
        sys.exit(1)
    if args.strict_two_stage and args.prompt_style == "strict_json":
        print("  Error: --strict-two-stage is only supported with --prompt-style strict.", file=sys.stderr)
        sys.exit(1)

    if args.parse_retry_max > 0 and not (
        args.strict_two_stage or args.prompt_style in ("strict", "strict_json")
    ):
        print(
            "  Warning: --parse-retry-max ignored unless --prompt-style strict|strict_json or --strict-two-stage.",
            file=sys.stderr,
        )
    if args.parse_retry_max > 0 and not args.parse_strict_output:
        print(
            "  Note: --parse-retry-max uses internal parsing; add --parse-strict-output to log parsed fields in JSONL.",
            file=sys.stderr,
        )

    print(f"  Primary checkpoint: {pt_path}")
    completions_a, retry_meta_a = run_for_checkpoint(
        news_a,
        meta_a,
        base_name,
        probes,
        device,
        args.prompt_style,
        prompt_fn,
        max_new_tokens,
        do_sample,
        args.temperature,
        repetition_penalty=repetition_penalty,
        strict_two_stage=args.strict_two_stage,
        reason_only=args.reason_only,
        seq_primary=seq_primary,
        num_labels=int(meta_a["num_labels"]),
        parse_extra_retries=max(0, args.parse_retry_max),
        retry_parse_lenient=args.parse_retry_lenient,
        retry_temperature=max(args.parse_retry_temperature, 1e-5),
    )

    completions_b: Optional[List[str]] = None
    retry_meta_b: List[Dict[str, Any]] = []
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
        completions_b, retry_meta_b = run_for_checkpoint(
            news_b,
            meta_b,
            base_name,
            probes,
            device,
            args.prompt_style,
            prompt_fn,
            max_new_tokens,
            do_sample,
            args.temperature,
            repetition_penalty=repetition_penalty,
            strict_two_stage=args.strict_two_stage,
            reason_only=args.reason_only,
            seq_primary=seq_compare,
            num_labels=int(meta_b["num_labels"]),
            parse_extra_retries=max(0, args.parse_retry_max),
            retry_parse_lenient=args.parse_retry_lenient,
            retry_temperature=max(args.parse_retry_temperature, 1e-5),
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
            if retry_meta_a[i]:
                row.update(retry_meta_a[i])
            if args.parse_strict_output:
                if args.prompt_style == "strict_json" and not args.strict_two_stage:
                    pc, pr, fv, fvs = parse_strict_json_completion(completions_a[i])
                else:
                    pc, pr, fv, fvs = parse_strict_completion(completions_a[i])
                row["parsed_category"] = pc
                row["parsed_reason"] = pr
                row["format_valid"] = fv
                row["format_valid_strict"] = fvs
                if p.get("gold_category") is not None:
                    row["matches_gold_category"] = bool(fv and pc is not None and pc == p["gold_category"])
                    row["matches_gold_category_strict"] = bool(
                        fvs and pc is not None and pc == p["gold_category"]
                    )
            if completions_b is not None:
                row["completion_compare"] = completions_b[i]
                if i < len(retry_meta_b) and retry_meta_b[i]:
                    for rk, rv in retry_meta_b[i].items():
                        row[f"{rk}_compare"] = rv
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

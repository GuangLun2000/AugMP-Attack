# IoA-Attack-GRMP

## File Structure

```
├── README.md                       # Project documentation (this file)
├── requirements.txt                # Dependencies for the project
├── main.py                         # Main experiment script: configures and runs FL 
├── fed_checkpoint.py               # Save global NewsClassifierModel after FL (optional)
├── decoder_adapters.py             # Pluggable SeqCLS → CausalLM backbone transfer
├── run_downstream_generation.py    # CLI: load Fed checkpoint, generate on probe JSON
├── client.py                       # Client logic (BenignClient, AttackerClient/GRMP)
├── server.py                       # Server: model aggregation and evaluation
├── models.py                       # Learning model definitions (NewsClassifierModel)
├── data_loader.py                  # Data loading (AG News, IMDB, DBpedia, Yahoo Answers)
├── data/financial_probes.json      # 10 finance-style probes (news + question) for downstream gen
├── data/ag_news_simple_probes.json # 10 short probes: AG News 4-class label + concise explanation (downstream)
├── scripts/sample_ag_business_probes.py  # Optional: sample 10 AG News Business rows into JSON
├── visualization.py                # Visualization module: generates figures
├── attack_baseline_alie.py         # ALIE attack baseline (NeurIPS '19)
├── attack_baseline_gaussian.py     # Gaussian attack baseline (USENIX Security '20)
├── attack_baseline_sign_flipping.py# Sign-flipping attack baseline (ICML '18)
├── GRMP_Attack_Colab.ipynb         # Google Colab notebook for interactive execution
├── AG_News_Datasets/               # AG News local data (train.csv, test.csv)
└── Yahoo_Answers_Datasets/         # Yahoo Answers local data (created on first run)
```

## Supported Models

Encoder-only (BERT-style): `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`  
Decoder-only (GPT-style): `gpt2`, `EleutherAI/pythia-160m`, `EleutherAI/pythia-1b`, `facebook/opt-125m`, `Qwen/Qwen2.5-0.5B`

Configure in `main.py` via `model_name`. Use base models (not Instruct) for classification fine-tuning.

## Supported Datasets

- **AG News**: `dataset='ag_news'`, `num_labels=4`, `max_length=128` (default)
- **IMDB** (stanfordnlp/imdb): `dataset='imdb'`, `num_labels=2`, `max_length=512` (or 256 for lower memory)
- **DBpedia 14** (fancyzhx/dbpedia_14): `dataset='dbpedia'`, `num_labels=14`, `max_length=512` (14 topic classes, 560K train / 70K test)
- **Yahoo Answers** (yassiracharki/Yahoo_Answers_10_categories_for_NLP): `dataset='yahoo_answers'`, `num_labels=10`, `max_length=256` (10 topic classes, 1.4M train / 60K test)

Configure in `main.py` via `dataset`, `num_labels`, and `max_length`.

**Note on dataset_size_limit**: When `dataset_size_limit` is set, both train and test are limited for faster experimentation: train uses up to `dataset_size_limit` samples, test uses up to `dataset_size_limit × 0.15` samples (same rule for all datasets).

**Local storage**: AG News and Yahoo Answers can be stored locally. AG News uses `AG_News_Datasets/`; Yahoo Answers uses `Yahoo_Answers_Datasets/`. On first run with Yahoo Answers, the dataset is downloaded from Hugging Face and saved to `Yahoo_Answers_Datasets/` for future use (same behavior as AG News).

## Install Dependencies

```python
!pip install -r requirements.txt
```

## Run the Code

### Local Execution

```bash
python main.py
```

### Save global model checkpoint (for downstream experiments)

After federated training, you can persist `server.global_model` (same `NewsClassifierModel` as during FL).

1. In `main.py` → `config`, set:
   - `save_global_checkpoint`: `True`
   - `global_checkpoint_subdir`: optional subfolder under `results/` (default: `global_checkpoint`). Use a unique name per run if multiple experiments share `results/`.
2. Re-run `python main.py`. Outputs:
   - `results/<global_checkpoint_subdir>/global_model.pt` — `state_dict` + embedded `metadata`
   - `results/<global_checkpoint_subdir>/checkpoint_metadata.json` — `model_name`, `num_labels`, `use_lora`, LoRA hyperparameters if applicable
   - If `use_lora=True`: `results/<global_checkpoint_subdir>/peft_adapter/` — PEFT `save_pretrained` (best-effort)

**FedLLM checkpoint examples:** set `model_name` to `EleutherAI/pythia-160m` or `Qwen/Qwen2.5-0.5B`, with `num_labels` / `dataset` consistent (e.g. AG News: `num_labels=4`), then enable `save_global_checkpoint` as above.

### Downstream causal generation (probe JSON)

Second-stage script loads the Fed **sequence-classification** checkpoint, copies the shared **backbone** into **`AutoModelForCausalLM`** (same HF `model_name`), keeps the **pretrained `lm_head`**, and runs `generate()` on a fixed probe list (no extra training).

**Adapters** ([`decoder_adapters.py`](decoder_adapters.py)): **Qwen2 / Qwen2.5** (`model.*`) and **Pythia / GPT-NeoX** (`gpt_neox.*`).

**Probes**

- [`data/ag_news_simple_probes.json`](data/ag_news_simple_probes.json) — short news + instruction to output **(1)** exactly one of World / Sports / Business / Sci/Tech and **(2)** one concise explanation sentence (aligns with AG News `num_labels=4`). Recommended with **`--prompt-style simple`** and **`--stable`** (`--stable` uses `max_new_tokens=64` by default to fit label + brief rationale).
- [`data/financial_probes.json`](data/financial_probes.json) — finance-themed synthetic snippets (harder for small base LMs).

To sample **AG News Business** lines into JSON, run:

```bash
python scripts/sample_ag_business_probes.py --csv AG_News_Datasets/train.csv -o data/financial_probes_ag.json
```

(AG CSV labels: `3` = Business.)

**Stability flags**

- **`--stable`**: greedy decoding, `max_new_tokens=64` by default (override with `--max-new-tokens`; sized for one class word + short explanation), `repetition_penalty=1.1` unless you set `--repetition-penalty`.
- **`--write-seq-cls-argmax`**: adds `seq_cls_label_id` / `seq_cls_label` from the **SeqCLS head** (same objective as Fed training) for a reproducible label column next to free-form `completion_*`.
- **`--prompt-style {default,simple}`**: `simple` uses a short `News: ... / Answer:` template; `default` uses `### News` / `### Task` / `### Answer`.

**Qwen2.5 + AG News example** (after saving a checkpoint with `model_name=Qwen/Qwen2.5-0.5B`, `dataset=ag_news`, `num_labels=4`):

```bash
python run_downstream_generation.py \
  --checkpoint results/global_checkpoint \
  --probes data/ag_news_simple_probes.json \
  --output results/downstream_gen.jsonl \
  --stable \
  --write-seq-cls-argmax \
  --prompt-style simple
```

**Single checkpoint** (legacy defaults: sampling, up to 128 new tokens):

```bash
python run_downstream_generation.py \
  --checkpoint results/global_checkpoint \
  --probes data/financial_probes.json \
  --output results/downstream_gen.jsonl \
  --max-new-tokens 128
```

**Paired clean vs poisoned** (same probes, two checkpoints):

```bash
python run_downstream_generation.py \
  --checkpoint results/global_checkpoint \
  --compare-checkpoint results_baseline/global_checkpoint \
  --probes data/ag_news_simple_probes.json \
  --output results/downstream_compare.jsonl \
  --stable \
  --write-seq-cls-argmax \
  --prompt-style simple
```

JSONL fields: `probe_id`, `news_text`, `question`, `prompt_style`, `completion_primary`; with `--compare-checkpoint`, also `completion_compare` and `seq_cls_compare_*`; with `--write-seq-cls-argmax`, also `seq_cls_label_id`, `seq_cls_label`. Use `--greedy` for greedy decoding without full `--stable`; `--base-model` overrides the HF id for CausalLM/tokenizer (only if it matches the saved architecture).

### Adding another decoder family

1. Subclass `DecoderAdapter` in [`decoder_adapters.py`](decoder_adapters.py) and implement `matches(model_name)` and `transfer_backbone(seq_cls_inner, causal_lm)`.
2. Append the class to `ADAPTER_REGISTRY` (list order = match priority).
3. Run `run_downstream_generation.py` with checkpoints trained on that `model_name`.

### Google Colab Execution

**Option 1: Simple Version (Recommended for quick runs)**
```python
# Cell 1: Install dependencies
!git clone https://github.com/GuangLun2000/IoA-Attack-GRMP.git
!pip install -r ./IoA-Attack-GRMP/requirements.txt

# Cell 2: Run experiment

!cd ./IoA-Attack-GRMP && python main.py
```

**Option 2: Interactive Notebook (Recommended for configuration changes)**
1. Open `GRMP_Attack_Colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells: Runtime → Run all

## Citation

If this repository has been helpful to you, please consider citing our work in your paper. Thank you so much!

```latex
@article{cai2025graph,
  title={Graph Representation-based Model Poisoning on the Heterogeneous Internet of Agents},
  author={Cai, Hanlin and Wang, Houtianfu and Dong, Haofan and Li, Kai and Akan, Ozgur B},
  journal={arXiv preprint arXiv:2511.07176},
  year={2025}
}
```
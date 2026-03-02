# IoA-Attack-GRMP

## File Structure

```
├── README.md                       # Project documentation (this file)
├── requirements.txt                # Dependencies for the project
├── main.py                         # Main experiment script: configures and runs FL 
├── client.py                       # Client logic (BenignClient, AttackerClient/GRMP)
├── server.py                       # Server: model aggregation and evaluation
├── models.py                       # Learning model definitions (NewsClassifierModel)
├── data_loader.py                  # Data loading (AG News, IMDB, DBpedia, Yahoo Answers)
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
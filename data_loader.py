# data_loader.py
# Data loader for text classification (AG News, IMDB, DBpedia, Yahoo Answers) for federated experiments.
# Note: data-agnostic attack setting — no training-time label flipping is performed.

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import urllib.request
import io
import os
from typing import List, Dict

class NewsDataset(Dataset):
    """Custom Dataset for text classification (AG News, IMDB, DBpedia, Yahoo Answers, etc.)"""

    def __init__(self, texts, labels, tokenizer, max_length=128,
                include_target_mask: bool = False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_target_mask = include_target_mask

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        return item





class DataManager:
    """Manages text classification data distribution (AG News, IMDB, DBpedia, Yahoo Answers) for federated experiments (data-agnostic attack setting)"""

    def __init__(self, num_clients, num_attackers, test_seed,
                 dataset_size_limit=None, batch_size=None, test_batch_size=None,
                 model_name: str = "distilbert-base-uncased", max_length: int = 128,
                 dataset: str = "ag_news"):
        
        """
        Initialize DataManager.
        
        Args:
            num_clients: Number of federated learning clients (required)
            num_attackers: Number of attacker clients (required)
            test_seed: Random seed for test sampling (required)
            dataset_size_limit: Limit dataset size (None = full dataset). For paper reproduction, use None.
                               When set, only limits training set; test set remains full for fair evaluation.
            batch_size: Batch size for training data loaders (required)
            test_batch_size: Batch size for test/validation data loaders (required)
            model_name: Hugging Face model name for tokenizer initialization
            max_length: Max token length (AG News: 128, IMDB: 256-512, DBpedia: 512, Yahoo Answers: 256)
            dataset: 'ag_news' | 'imdb' | 'dbpedia' | 'yahoo_answers'
        """

        if batch_size is None or test_batch_size is None:
            raise ValueError("batch_size and test_batch_size must be provided via config (see main.py).")

        self.num_clients = num_clients
        self.num_attackers = num_attackers
        self.test_seed = test_seed
        self.dataset_size_limit = dataset_size_limit
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.max_length = max_length
        self.model_name = model_name
        self.dataset = dataset.lower()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle padding for decoder-only models (GPT-style)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"  📝 Set pad_token = eos_token ('{self.tokenizer.eos_token}') for {model_name}")
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print(f"  📝 Added new pad_token '[PAD]' for {model_name}")

        if self.dataset == "imdb":
            print("Loading IMDB dataset (stanfordnlp/imdb)...")
        elif self.dataset == "dbpedia":
            print("Loading DBpedia dataset (fancyzhx/dbpedia_14)...")
        elif self.dataset == "yahoo_answers":
            print("Loading Yahoo Answers dataset (yassiracharki/Yahoo_Answers_10_categories_for_NLP)...")
        else:
            print("Loading AG News dataset...")
        self._load_data()

    def _load_data(self):
        """Dispatch to dataset-specific loader."""
        if self.dataset == "imdb":
            self._load_imdb()
        elif self.dataset == "dbpedia":
            self._load_dbpedia()
        elif self.dataset == "yahoo_answers":
            self._load_yahoo_answers()
        else:
            self._load_ag_news()

    def _load_imdb(self):
        """Load IMDB dataset from Hugging Face (stanfordnlp/imdb)."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("IMDB requires datasets library. Install: pip install datasets")

        ds = load_dataset("stanfordnlp/imdb")
        train_data = ds["train"]
        test_data = ds["test"]

        self.train_texts = [str(x) for x in train_data["text"]]
        self.train_labels = list(train_data["label"])
        self.test_texts = [str(x) for x in test_data["text"]]
        self.test_labels = list(test_data["label"])

        print(f"  📊 Full IMDB Dataset: Train={len(self.train_texts)}, Test={len(self.test_texts)}")

        if self.dataset_size_limit is not None and self.dataset_size_limit > 0:
            rng = np.random.default_rng(42)
            n_train = min(self.dataset_size_limit, len(self.train_texts))
            n_test = min(int(self.dataset_size_limit * 0.15), len(self.test_texts))
            idx_train = rng.choice(len(self.train_texts), n_train, replace=False)
            idx_test = rng.choice(len(self.test_texts), n_test, replace=False)
            self.train_texts = [self.train_texts[i] for i in idx_train]
            self.train_labels = [self.train_labels[i] for i in idx_train]
            self.test_texts = [self.test_texts[i] for i in idx_test]
            self.test_labels = [self.test_labels[i] for i in idx_test]
            print(f"  ⚠️  Using limited size: Train={len(self.train_texts)}, Test={len(self.test_texts)} (test = train_limit × 0.15)")

        print(f"  ✅ IMDB ready! Train: {len(self.train_texts)}, Test: {len(self.test_texts)}")

    def _load_dbpedia(self):
        """Load DBpedia 14 dataset from Hugging Face (fancyzhx/dbpedia_14)."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("DBpedia requires datasets library. Install: pip install datasets")

        ds = load_dataset("fancyzhx/dbpedia_14")
        train_data = ds["train"]
        test_data = ds["test"]

        # DBpedia has 'title' and 'content' fields; combine them like AG News
        train_texts_combined = [f"{str(title)} {str(content)}" for title, content in zip(train_data["title"], train_data["content"])]
        test_texts_combined = [f"{str(title)} {str(content)}" for title, content in zip(test_data["title"], test_data["content"])]

        self.train_texts = train_texts_combined
        self.train_labels = list(train_data["label"])
        self.test_texts = test_texts_combined
        self.test_labels = list(test_data["label"])

        print(f"  📊 Full DBpedia Dataset: Train={len(self.train_texts)}, Test={len(self.test_texts)}")

        if self.dataset_size_limit is not None and self.dataset_size_limit > 0:
            rng = np.random.default_rng(42)
            n_train = min(self.dataset_size_limit, len(self.train_texts))
            n_test = min(int(self.dataset_size_limit * 0.15), len(self.test_texts))
            idx_train = rng.choice(len(self.train_texts), n_train, replace=False)
            idx_test = rng.choice(len(self.test_texts), n_test, replace=False)
            self.train_texts = [self.train_texts[i] for i in idx_train]
            self.train_labels = [self.train_labels[i] for i in idx_train]
            self.test_texts = [self.test_texts[i] for i in idx_test]
            self.test_labels = [self.test_labels[i] for i in idx_test]
            print(f"  ⚠️  Using limited size: Train={len(self.train_texts)}, Test={len(self.test_texts)} (test = train_limit × 0.15)")

        print(f"  ✅ DBpedia ready! Train: {len(self.train_texts)}, Test: {len(self.test_texts)}")

    def _load_yahoo_answers(self):
        """
        Load Yahoo Answers 10-category dataset.
        1. Check Yahoo_Answers_Datasets/ directory first (like AG News).
        2. If not found, download from Hugging Face and save to local for next time.
        """
        data_dir = 'Yahoo_Answers_Datasets'
        train_file = os.path.join(data_dir, 'train.csv')
        test_file = os.path.join(data_dir, 'test.csv')

        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"  ✅ Found local data files in {data_dir}/ directory. Loading...")
            train_df = pd.read_csv(train_file, header=None, names=['label', 'text'], quoting=1)
            test_df = pd.read_csv(test_file, header=None, names=['label', 'text'], quoting=1)
            self.train_texts = train_df['text'].fillna('').astype(str).tolist()
            self.train_labels = [(int(x) - 1) for x in train_df['label']]
            self.test_texts = test_df['text'].fillna('').astype(str).tolist()
            self.test_labels = [(int(x) - 1) for x in test_df['label']]
        else:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("Yahoo Answers requires datasets library. Install: pip install datasets")

            print(f"  🌐 Local data not found. Downloading from Hugging Face...")
            ds = load_dataset("yassiracharki/Yahoo_Answers_10_categories_for_NLP")
            train_data = ds["train"]
            test_data = ds["test"]

            cols = train_data.column_names
            def _get_col(candidates):
                for c in candidates:
                    if c in cols:
                        return c
                return None
            label_col = _get_col(["class_index", "Class Index", "label"]) or cols[0]
            title_col = _get_col(["question_title", "Question Title"]) or cols[1]
            content_col = _get_col(["question_content", "Question Content"]) or cols[2]
            answer_col = _get_col(["best_answer", "Best Answer"]) or (cols[3] if len(cols) > 3 else None)

            def _combine_text(t, c, a):
                parts = [str(x or "").strip() for x in [t, c, a] if x is not None]
                return " ".join(p for p in parts if p) or " "

            if answer_col:
                train_texts = [_combine_text(t, c, a) for t, c, a in zip(train_data[title_col], train_data[content_col], train_data[answer_col])]
                test_texts = [_combine_text(t, c, a) for t, c, a in zip(test_data[title_col], test_data[content_col], test_data[answer_col])]
            else:
                train_texts = [_combine_text(t, c, None) for t, c in zip(train_data[title_col], train_data[content_col])]
                test_texts = [_combine_text(t, c, None) for t, c in zip(test_data[title_col], test_data[content_col])]
            train_labels_raw = list(train_data[label_col])
            test_labels_raw = list(test_data[label_col])

            self.train_texts = train_texts
            self.train_labels = [int(x) - 1 for x in train_labels_raw]
            self.test_texts = test_texts
            self.test_labels = [int(x) - 1 for x in test_labels_raw]

            # Save to local for next time (like AG News)
            os.makedirs(data_dir, exist_ok=True)
            train_save = pd.DataFrame({'label': [l + 1 for l in self.train_labels], 'text': self.train_texts})
            test_save = pd.DataFrame({'label': [l + 1 for l in self.test_labels], 'text': self.test_texts})
            train_save.to_csv(train_file, index=False, header=False, quoting=1)
            test_save.to_csv(test_file, index=False, header=False, quoting=1)
            print(f"  ✅ Saved to {data_dir}/ for future use.")

        print(f"  📊 Full Yahoo Answers Dataset: Train={len(self.train_texts)}, Test={len(self.test_texts)}")

        if self.dataset_size_limit is not None and self.dataset_size_limit > 0:
            rng = np.random.default_rng(42)
            n_train = min(self.dataset_size_limit, len(self.train_texts))
            n_test = min(int(self.dataset_size_limit * 0.15), len(self.test_texts))
            idx_train = rng.choice(len(self.train_texts), n_train, replace=False)
            idx_test = rng.choice(len(self.test_texts), n_test, replace=False)
            self.train_texts = [self.train_texts[i] for i in idx_train]
            self.train_labels = [self.train_labels[i] for i in idx_train]
            self.test_texts = [self.test_texts[i] for i in idx_test]
            self.test_labels = [self.test_labels[i] for i in idx_test]
            print(f"  ⚠️  Using limited size: Train={len(self.train_texts)}, Test={len(self.test_texts)} (test = train_limit × 0.15)")

        print(f"  ✅ Yahoo Answers ready! Train: {len(self.train_texts)}, Test: {len(self.test_texts)}")

    def _load_ag_news(self):
        """
        [OPTIMIZED] Robust data loading with local cache priority.
        1. Check AG_News_Datasets/ directory first (if exists).
        2. Check root directory for train.csv/test.csv.
        3. If not found, download from GitHub.
        4. Strict failure if download fails (No synthetic data).
        """
        # Priority 1: Check AG_News_Datasets/ directory
        train_file_alt = 'AG_News_Datasets/train.csv'
        test_file_alt = 'AG_News_Datasets/test.csv'
        # Priority 2: Check root directory
        train_file = 'train.csv'
        test_file = 'test.csv'
        
        # URLs (verified: mhjabreel/CharCnn_Keras is a reliable source)
        train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
        test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"

        try:
            # 1. Try Local Load (Priority 1: AG_News_Datasets/ directory)
            if os.path.exists(train_file_alt) and os.path.exists(test_file_alt):
                print(f"  ✅ Found local data files in AG_News_Datasets/ directory. Loading...")
                train_df = pd.read_csv(train_file_alt, header=None, names=['label', 'title', 'text'])
                test_df = pd.read_csv(test_file_alt, header=None, names=['label', 'title', 'text'])
            # Priority 2: Check root directory
            elif os.path.exists(train_file) and os.path.exists(test_file):
                print(f"  ✅ Found local data files ({train_file}, {test_file}). Loading...")
                train_df = pd.read_csv(train_file, header=None, names=['label', 'title', 'text'])
                test_df = pd.read_csv(test_file, header=None, names=['label', 'title', 'text'])
            
            # 2. Try Download
            else:
                print("  🌐 Local data not found. Downloading from GitHub...")
                print(f"     Source: {train_url}")
                
                # Train Data
                with urllib.request.urlopen(train_url, timeout=20) as response:
                    data = response.read().decode('utf-8')
                    # Save to root directory for next time
                    with open(train_file, 'w', encoding='utf-8') as f:
                        f.write(data)
                    train_df = pd.read_csv(io.StringIO(data), header=None, names=['label', 'title', 'text'])
                
                # Test Data
                with urllib.request.urlopen(test_url, timeout=20) as response:
                    data = response.read().decode('utf-8')
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write(data)
                    test_df = pd.read_csv(io.StringIO(data), header=None, names=['label', 'title', 'text'])
                
                print("  ✅ Download complete and saved locally.")

        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: Data loading failed: {e}")
            print("🛑 STRICT MODE: Synthetic data generation is DISABLED to ensure validity.")
            print("   Please ensure internet access or manually place 'train.csv' and 'test.csv' in the folder.")
            raise e

        # Process Data
        # Combine title and text
        train_df['full_text'] = train_df['title'].astype(str) + ' ' + train_df['text'].astype(str)
        test_df['full_text'] = test_df['title'].astype(str) + ' ' + test_df['text'].astype(str)

        # Adjust labels 1-4 -> 0-3
        train_df['label'] = train_df['label'] - 1
        test_df['label'] = test_df['label'] - 1

        # Print full dataset size
        print(f"  📊 Full AG News Dataset: Train={len(train_df)}, Test={len(test_df)}")
        
        # Use full dataset by default
        # AG News full dataset: ~120,000 training samples, ~7,600 test samples
        # If dataset_size_limit is set, use it for faster experimentation (not recommended for paper reproduction)
        if hasattr(self, 'dataset_size_limit') and self.dataset_size_limit is not None:
            if self.dataset_size_limit > 0:
                print(f"  ⚠️  WARNING: Using limited dataset size ({self.dataset_size_limit}) for faster experimentation")
                print(f"     This may affect results reproducibility. For paper reproduction, use full dataset.")
                train_sample = train_df.sample(n=min(self.dataset_size_limit, len(train_df)), random_state=42)
                test_sample = test_df.sample(n=min(int(self.dataset_size_limit * 0.15), len(test_df)), random_state=42)
            else:
                # Use full dataset
                train_sample = train_df
                test_sample = test_df
        else:
            # Use full dataset (default, per paper)
            train_sample = train_df
            test_sample = test_df

        self.train_texts = train_sample['full_text'].tolist()
        self.train_labels = train_sample['label'].tolist()
        self.test_texts = test_sample['full_text'].tolist()
        self.test_labels = test_sample['label'].tolist()

        print(f"  ✅ Dataset ready! Train: {len(self.train_texts)}, Test: {len(self.test_texts)}")
        if len(self.train_texts) < len(train_df) or len(self.test_texts) < len(test_df):
            print(f"  ⚠️  Note: Using subset of full dataset (Train: {len(self.train_texts)}/{len(train_df)}, "
                  f"Test: {len(self.test_texts)}/{len(test_df)})")
        else:
            print(f"  ✅ Using FULL AG News dataset (per paper requirements)")

    def get_empty_loader(self) -> DataLoader:
        """Return an empty loader for data-agnostic attackers."""
        return DataLoader(NewsDataset([], [], self.tokenizer, max_length=self.max_length), batch_size=self.batch_size, shuffle=False)

    def get_proxy_eval_loader(self, sample_size: int = 128) -> DataLoader:
        """
        Small clean proxy set for attacker-side F(w'_g) estimation.
        Uses a deterministic subset of the test set (no label flips).
        """
        if not self.test_texts:
            return self.get_empty_loader()
        rng = np.random.default_rng(self.test_seed)
        idx = rng.choice(len(self.test_texts), size=min(sample_size, len(self.test_texts)), replace=False)
        proxy_texts = [self.test_texts[i] for i in idx]
        proxy_labels = [self.test_labels[i] for i in idx]
        dataset = NewsDataset(proxy_texts, proxy_labels, self.tokenizer, max_length=self.max_length)
        return DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False)

    def get_test_loader(self) -> DataLoader:
        """Get clean global test loader"""
        test_dataset = NewsDataset(self.test_texts, self.test_labels, self.tokenizer, max_length=self.max_length)
        return DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)


"""Load ESCAPE and GenPept-Curated-2025 AMP datasets."""

import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

LABELS = ["antibacterial", "antifungal", "antiviral", "antiparasitic"]

def load_escape() -> DatasetDict:
    """Load ESCAPE folds into HuggingFace DatasetDict."""
    fold1 = pd.read_csv(DATA_DIR / "Fold1.tab", sep="\t")
    fold2 = pd.read_csv(DATA_DIR / "Fold2.tab", sep="\t")
    test = pd.read_csv(DATA_DIR / "Test.tab", sep="\t")
    return DatasetDict({
        "fold1": Dataset.from_pandas(fold1),
        "fold2": Dataset.from_pandas(fold2),
        "test": Dataset.from_pandas(test),
    })

def load_genpept() -> Dataset:
    """Load GenPept-Curated-2025 balanced dataset."""
    df = pd.read_csv(DATA_DIR / "GenPept-Curated-2025" / "data" / "balanced_11000.csv")
    return Dataset.from_pandas(df)

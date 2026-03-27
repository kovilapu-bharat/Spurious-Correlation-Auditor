"""
Data Input Module
-----------------
Handles dataset loading (built-in & CSV upload), validation, preprocessing,
and train/test splitting.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Built-in dataset registry
# ---------------------------------------------------------------------------

BUILTIN_DATASETS: dict[str, dict] = {
    "Heart Disease (UCI)": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "columns": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
        ],
        "target": "target",
        "description": "UCI Heart Disease dataset — predict presence of heart disease.",
    },
    "Breast Cancer (sklearn)": {
        "loader": load_breast_cancer,
        "description": "Wisconsin Breast Cancer dataset — classify malignant vs. benign.",
    },
    "Diabetes (sklearn)": {
        "loader": load_diabetes,
        "description": "Diabetes progression dataset — regression target (will be binarised).",
    },
    "Iris (sklearn)": {
        "loader": load_iris,
        "description": "Iris flower dataset — classify 3 species.",
    },
    "Waterbirds (Spurious)": {
        "loader_fn": "_load_waterbirds",
        "description": (
            "Stanford Waterbirds dataset — classify waterbird vs landbird. "
            "Contains a known spurious correlation between background type "
            "(water/land) and bird label."
        ),
    },
    "IMDB Sentiment": {
        "loader_fn": "_load_imdb",
        "description": (
            "Stanford IMDB Large Movie Review Dataset — classify movie reviews "
            "as positive or negative sentiment. Text is converted to TF-IDF "
            "features for tabular analysis."
        ),
    },
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class AuditData:
    """Container holding everything downstream modules need."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str
    df_full: pd.DataFrame  # original (unscaled) dataframe for EDA
    class_names: list[str] = field(default_factory=list)
    scaler: Optional[StandardScaler] = None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_heart_disease() -> pd.DataFrame:
    """Download and prepare the Heart Disease dataset."""
    info = BUILTIN_DATASETS["Heart Disease (UCI)"]
    df = pd.read_csv(info["url"], header=None, names=info["columns"], na_values="?")
    df.dropna(inplace=True)
    # Binarise target: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)
    return df


def _load_sklearn_dataset(name: str) -> pd.DataFrame:
    """Load a scikit-learn built-in dataset and return a DataFrame."""
    info = BUILTIN_DATASETS[name]
    bunch = info["loader"]()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)

    if name == "Diabetes (sklearn)":
        # Binarise for classification: above-median → 1
        median_val = np.median(bunch.target)
        df["target"] = (bunch.target > median_val).astype(int)
    else:
        df["target"] = bunch.target

    return df


def _load_waterbirds(extract_images: bool = False, max_samples: int | None = 400) -> pd.DataFrame:
    """
    Load the Stanford Waterbirds dataset.

    Creates a tabular classification dataset.
    If extract_images is False:
      Returns metadata features (species_id, place) vs target.
    If extract_images is True:
      Loads actual images, runs them through a pretrained ResNet50,
      and returns the 2048-dimensional embeddings as features.
    """
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data" / "waterbirds_v1.0"
    meta_path = data_dir / "metadata.csv"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Waterbirds metadata not found at {meta_path}. "
            f"Run 'python download_waterbirds_samples.py' first."
        )

    meta = pd.read_csv(meta_path)

    # Subsample if necessary
    if max_samples and len(meta) > max_samples:
        import warnings
        warnings.warn(f"Subsampling Waterbirds dataset to {max_samples} images to save time.")
        meta = meta.sample(n=max_samples, random_state=42).reset_index(drop=True)

    meta["species_id"] = meta["img_filename"].str.extract(r"^(\d+)\.").astype(int)

    if not extract_images:
        df = pd.DataFrame({
            "species_id": meta["species_id"].values,
            "place": meta["place"].values,
            "target": meta["y"].values,
        })
        return df

    # --- IMAGE PIPELINE: ResNet50 Feature Extraction ---
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image

    print("Loading pretrained ResNet50 model for feature extraction...")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    # Remove the final classification layer to get 2048-d features
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class WaterbirdsDataset(Dataset):
        def __init__(self, meta_df, data_dir, transform):
            self.meta = meta_df
            self.data_dir = data_dir
            self.transform = transform
            
        def __len__(self):
            return len(self.meta)
            
        def __getitem__(self, idx):
            img_path = self.data_dir / self.meta.iloc[idx]["img_filename"]
            if not img_path.exists():
                return torch.zeros((3, 224, 224))
            img = Image.open(img_path).convert("RGB")
            return self.transform(img)

    dataset = WaterbirdsDataset(meta, data_dir, preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    features_list = []
    print(f"Extracting features for {len(meta)} images (Batched)...")
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch)
            features_list.append(out.cpu().numpy().reshape(batch.size(0), -1))

    features_array = np.vstack(features_list)
    
    # Build dataframe
    feature_cols = [f"resnet_{i}" for i in range(2048)]
    df = pd.DataFrame(features_array, columns=feature_cols)
    df["target"] = meta["y"].values
    
    # Store the actual image paths and true background for later UI use
    df["_img_path"] = meta["img_filename"].values
    df["_true_place"] = meta["place"].values
    
    return df


def _load_imdb(max_features: int = 500, max_samples: int | None = None) -> pd.DataFrame:
    """
    Load the Stanford IMDB sentiment dataset from local extracted files.

    Reads plain-text review files from data/aclImdb/train/ and data/aclImdb/test/,
    then converts them to a tabular dataset using TF-IDF vectorization.

    Parameters
    ----------
    max_features : int
        Number of TF-IDF features to extract (vocabulary size).
    max_samples : int | None
        If set, limit the total number of reviews loaded (balanced pos/neg).
        Useful for faster prototyping.

    Returns
    -------
    pd.DataFrame
        DataFrame with TF-IDF feature columns + 'target' column
        (0 = negative, 1 = positive).
    """
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data" / "aclImdb"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"IMDB dataset not found at {data_dir}. "
            f"Run 'python download_imdb.py' first."
        )

    texts: list[str] = []
    labels: list[int] = []

    # Read from both train and test directories
    for split in ("train", "test"):
        for label_name, label_val in (("neg", 0), ("pos", 1)):
            folder = data_dir / split / label_name
            if not folder.exists():
                continue
            files = sorted(folder.glob("*.txt"))
            if max_samples is not None:
                # Take equal samples from each category
                per_category = max_samples // 4
                files = files[:per_category]
            for fpath in files:
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                    texts.append(text)
                    labels.append(label_val)
                except Exception:
                    continue

    if not texts:
        raise FileNotFoundError(
            f"No review files found in {data_dir}. "
            f"Ensure the dataset is fully extracted."
        )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = [f"tfidf_{w}" for w in vectorizer.get_feature_names_out()]

    df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
    )
    df["target"] = labels

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_builtin_dataset(name: str, **kwargs) -> pd.DataFrame:
    """Load a built-in dataset by registry name and return a DataFrame."""
    if name == "Heart Disease (UCI)":
        return _load_heart_disease()
    if name == "Waterbirds (Spurious)":
        return _load_waterbirds(**kwargs)
    if name == "IMDB Sentiment":
        return _load_imdb()
    return _load_sklearn_dataset(name)



def load_csv_dataset(uploaded_file, target_column: str) -> pd.DataFrame:
    """Read an uploaded CSV file into a DataFrame, validating the target."""
    df = pd.read_csv(uploaded_file)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the CSV.")
    return df


def prepare_data(
    df: pd.DataFrame,
    target_column: str = "target",
    test_size: float = 0.25,
    scale: bool = True,
    random_state: int = 42,
) -> AuditData:
    """
    Clean, encode, split, and optionally scale a DataFrame.

    Returns an ``AuditData`` instance ready for downstream modules.
    """
    df = df.copy()

    # --- encode categorical columns ---
    label_encoders: dict[str, LabelEncoder] = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # --- separate features / target ---
    feature_cols = [c for c in df.columns if c != target_column]
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_column].values

    # Fill remaining NaNs with column median
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # --- split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if _is_classification(y) else None
    )

    # --- scale ---
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # --- class names ---
    unique_classes = sorted(np.unique(y))
    class_names = [str(c) for c in unique_classes]

    return AuditData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
        target_name=target_column,
        df_full=df,
        class_names=class_names,
        scaler=scaler,
    )


def _is_classification(y: np.ndarray, max_unique: int = 20) -> bool:
    """Heuristic: treat as classification if ≤ max_unique distinct values."""
    return len(np.unique(y)) <= max_unique

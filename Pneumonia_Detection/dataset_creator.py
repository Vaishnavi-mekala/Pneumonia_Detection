#!/usr/bin/env python3
"""
Create and prepare a chest X-ray dataset for pneumonia classification.
This script downloads a public dataset when available and builds a binary dataset.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PUBLIC_GITHUB_REPO = "https://github.com/ieee8023/covid-chestxray-dataset.git"
TEMP_REPO_DIR = Path(tempfile.gettempdir()) / "covid-chestxray-dataset"


def _download_public_dataset(repo_url: str = PUBLIC_GITHUB_REPO, dest: Path = TEMP_REPO_DIR) -> Path:
    """Clone the public GitHub dataset repository if needed."""
    if dest.exists() and (dest / "metadata.csv").exists():
        return dest

    if dest.exists():
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(dest)])
    return dest


def _label_row(row):
    finding = str(row.get("finding", "")).strip()
    if finding == "No Finding":
        return "NORMAL"
    if finding.startswith("Pneumonia"):
        return "PNEUMONIA"
    if finding == "Pneumonia/Viral/COVID-19":
        return "PNEUMONIA"
    return None


def create_public_dataset(base_path: str = "dataset/chest_xray", max_per_class: int = 600):
    """Download the public dataset and create a train/val split for pneumonia detection."""
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)

    src_dir = _download_public_dataset()
    metadata_path = src_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found at {metadata_path}")

    df = pd.read_csv(metadata_path)
    df = df.dropna(subset=["finding", "folder", "filename"])
    df["label"] = df.apply(_label_row, axis=1)
    df = df[df["label"].isin(["NORMAL", "PNEUMONIA"])]

    if df.empty:
        raise ValueError("No labeled data found for NORMAL/PNEUMONIA in the public dataset.")

    selected_rows = []
    for label in ["NORMAL", "PNEUMONIA"]:
        class_df = df[df["label"] == label]
        max_items = min(len(class_df), max_per_class)
        selected_rows.append(class_df.sample(n=max_items, random_state=42))
    df = pd.concat(selected_rows).reset_index(drop=True)

    for split in ["train", "val"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            target_dir = base_dir / split / label
            target_dir.mkdir(parents=True, exist_ok=True)

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    norm_df = df[df["label"] == "NORMAL"]
    pneu_df = df[df["label"] == "PNEUMONIA"]

    def copy_rows(rows, target_split):
        for _, row in rows.iterrows():
            folder = str(row["folder"]).strip()
            filename = str(row["filename"]).strip()
            src_path = src_dir / folder / filename
            if not src_path.exists():
                continue
            target_path = base_dir / target_split / row["label"] / filename
            try:
                with Image.open(src_path) as img:
                    img = img.convert("L")
                    img = img.resize((224, 224), Image.LANCZOS)
                    img.save(target_path, format="JPEG", quality=90)
            except Exception:
                shutil.copy2(src_path, target_path)

    train_norm = norm_df.groupby("label").head(int(len(norm_df) * 0.8))
    val_norm = norm_df.drop(train_norm.index)
    train_pneu = pneu_df.groupby("label").head(int(len(pneu_df) * 0.8))
    val_pneu = pneu_df.drop(train_pneu.index)

    copy_rows(train_norm, "train")
    copy_rows(val_norm, "val")
    copy_rows(train_pneu, "train")
    copy_rows(val_pneu, "val")

    train_count = sum(len(list((base_dir / "train" / label).glob("*.jpg"))) + len(list((base_dir / "train" / label).glob("*.jpeg"))) for label in ["NORMAL", "PNEUMONIA"])
    val_count = sum(len(list((base_dir / "val" / label).glob("*.jpg"))) + len(list((base_dir / "val" / label).glob("*.jpeg"))) for label in ["NORMAL", "PNEUMONIA"])

    print(f"Created public dataset at {base_dir} with {train_count} train and {val_count} validation images.")
    return True


def create_synthetic_dataset(base_path="dataset/chest_xray", num_per_class=50):
    """Create synthetic chest X-ray images for training/validation."""
    
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    labels = ["NORMAL", "PNEUMONIA"]
    splits = ["train", "val"]
    
    np.random.seed(42)
    
    for split in splits:
        for label in labels:
            dir_path = base_dir / split / label
            dir_path.mkdir(parents=True, exist_ok=True)
            
            num_images = num_per_class if split == "train" else max(1, num_per_class // 5)
            
            for i in range(num_images):
                img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
                
                if label == "PNEUMONIA":
                    y, x = np.ogrid[:224, :224]
                    mask = ((x - 112)**2 + (y - 112)**2) < 2000
                    img_array[mask] = np.clip(img_array[mask] + np.random.randint(30, 80), 0, 255)
                else:
                    y, x = np.ogrid[:224, :224]
                    gradient = (x / 224.0 * 100 + y / 224.0 * 100).astype(np.uint8)
                    img_array = np.clip(img_array + gradient, 0, 255).astype(np.uint8)
                
                img = Image.fromarray(img_array, mode='L')
                img_path = dir_path / f"image_{i:04d}.jpeg"
                img.save(img_path)
            
            print(f"Created {num_images} {label} images in {split}/")
    
    print(f"Synthetic dataset created at {base_path}")
    return True


if __name__ == "__main__":
    try:
        create_public_dataset()
    except Exception as e:
        print(f"Could not build public dataset: {e}\nFalling back to synthetic data.")
        create_synthetic_dataset()

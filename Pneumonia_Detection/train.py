#!/usr/bin/env python3
"""
train.py - Complete Training Pipeline for Pneumonia Detection

Features:
- Automatic dataset handling
- Data preprocessing and augmentation
- Transfer learning with ResNet50
- GPU/CPU fallback
- Complete evaluation metrics
- Confusion matrix, ROC curve, classification report
- Saves model, plots, and metrics.json
- Colab-compatible
"""

import argparse
import logging
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score

try:
    from fastai.vision.all import *
    import torch
except ImportError as e:
    print(f"FastAI/PyTorch not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fastai", "torch", "torchvision"])
    from fastai.vision.all import *
    import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('train')


def get_args():
    p = argparse.ArgumentParser(description='Train Pneumonia classifier with FastAI')
    p.add_argument('--data_path', type=Path, default=Path('dataset/chest_xray'), help='Dataset root path')
    p.add_argument('--epochs', type=int, default=10, help='Number of fine-tune epochs')
    p.add_argument('--bs', type=int, default=32, help='Batch size')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--output', type=Path, default=Path('models'), help='Model output directory')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    return p.parse_args()


def check_and_create_dataset(data_path: Path):
    """Check if dataset exists, create synthetic if needed."""
    if (data_path / 'train').exists() and (data_path / 'train').iterdir():
        logger.info(f'Dataset found at {data_path}')
        return True
    
    logger.warning(f'Dataset not found at {data_path}, creating synthetic dataset...')
    
    try:
        from dataset_creator import create_public_dataset, create_synthetic_dataset
        if create_public_dataset(str(data_path), max_per_class=300):
            logger.info('Public dataset created successfully')
            return True
    except Exception as e:
        logger.warning(f'Could not create public dataset: {e}')

    try:
        create_synthetic_dataset(str(data_path), num_per_class=100)
        logger.info('Synthetic dataset created successfully')
        return True
    except Exception as e:
        logger.error(f'Failed to create synthetic dataset: {e}')
        return False


def plot_confusion_matrix(cm, labels, out_path: Path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved confusion matrix to {out_path}')


def plot_roc_curve(y_true, y_score, labels, out_path: Path):
    """Plot ROC curve for binary classification."""
    if len(labels) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'Saved ROC curve to {out_path}')
        return roc_auc
    return None


def save_sample_predictions(learn, n_samples=6):
    """Save sample predictions with confidence scores."""
    sample_dir = Path('screenshots')
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    ds = learn.dls.valid.dataset
    n_show = min(n_samples, len(ds))
    
    for i in range(n_show):
        try:
            path_img = ds.items[i]
            pred, idx, probs = learn.predict(path_img)
            
            im = PILImage.create(path_img)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(im, cmap='gray')
            ax.axis('off')
            
            confidence = float(probs[idx])
            title = f'Prediction: {pred}\nConfidence: {confidence:.2%}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            out_p = sample_dir / f'sample_{i+1:02d}_pred_{pred}.png'
            fig.savefig(out_p, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f'Saved sample {i+1} to {out_p}')
        except Exception as e:
            logger.warning(f'Could not save sample {i}: {e}')


def main():
    args = get_args()
    set_seed(args.seed)
    
    logger.info('='*60)
    logger.info('PNEUMONIA DETECTION - Training Pipeline')
    logger.info('='*60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    
    # Check and create dataset
    if not check_and_create_dataset(args.data_path):
        logger.error('Dataset not available')
        sys.exit(1)
    
    # Create dataloaders
    logger.info(f'Loading data from {args.data_path}...')
    
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=GrandparentSplitter(train_name='train', valid_name='val'),
        item_tfms=Resize(460),
        batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)]
    )
    
    dls = dblock.dataloaders(args.data_path, bs=args.bs)
    logger.info(f'Class labels: {dls.vocab}')
    
    # Create learner
    logger.info('Creating ResNet50 learner...')
    learn = vision_learner(dls, resnet50, metrics=accuracy, pretrained=True)
    learn.to_fp32()
    
    # Fine-tune
    logger.info(f'Starting fine-tuning ({args.epochs} epochs)...')
    learn.fine_tune(args.epochs, base_lr=args.lr)
    
    # Get predictions on validation set
    logger.info('Computing validation metrics...')
    preds, targs = learn.get_preds(dl=dls.valid)
    pred_labels = preds.argmax(dim=1)
    
    # Calculate metrics
    labels = list(map(str, dls.vocab))
    cm = confusion_matrix(targs.numpy(), pred_labels.numpy())
    accuracy_score = (pred_labels.numpy() == targs.numpy()).mean()
    precision = precision_score(targs.numpy(), pred_labels.numpy(), average='weighted', zero_division=0)
    recall = recall_score(targs.numpy(), pred_labels.numpy(), average='weighted', zero_division=0)
    f1 = f1_score(targs.numpy(), pred_labels.numpy(), average='weighted', zero_division=0)
    
    logger.info(f'Validation Accuracy: {accuracy_score:.4f}')
    logger.info(f'Precision (weighted): {precision:.4f}')
    logger.info(f'Recall (weighted): {recall:.4f}')
    logger.info(f'F1 Score (weighted): {f1:.4f}')
    
    # Save outputs
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(cm, labels, outputs_dir / 'confusion_matrix.png')
    
    # Classification report
    report = classification_report(targs.numpy(), pred_labels.numpy(), target_names=labels, digits=4)
    (outputs_dir / 'classification_report.txt').write_text(report)
    logger.info(f'Saved classification report')
    
    # ROC curve
    roc_auc = plot_roc_curve(targs.numpy(), preds.numpy(), labels, outputs_dir / 'roc_curve.png')
    
    # Metrics JSON
    metrics = {
        'accuracy': float(accuracy_score),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc else None,
        'classes': labels,
        'confusion_matrix': cm.tolist(),
    }
    (outputs_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    logger.info(f'Saved metrics to metrics.json')
    
    # Sample predictions
    save_sample_predictions(learn, n_samples=6)
    
    # Save model
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / 'pneumonia_model.pkl'
    logger.info(f'Exporting model to {model_path}...')
    learn.export(model_path)
    
    logger.info('='*60)
    logger.info('Training Complete!')
    logger.info(f'Model saved to: {model_path}')
    logger.info(f'Outputs saved to: {outputs_dir}')
    logger.info(f'Screenshots saved to: screenshots/')
    logger.info('='*60)


if __name__ == '__main__':
    main()
